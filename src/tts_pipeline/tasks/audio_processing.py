"""
Audio processing tasks for TTS dataset creation
Includes Silero VAD-based silence removal and denoising
"""
import logging
import numpy as np
import librosa
from typing import Dict, List, Tuple
import torch

from ..config.settings import settings

logger = logging.getLogger(__name__)


def remove_long_silences_silero(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Remove long silences using Silero VAD"""
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        
        logger.info("Removing long silences using Silero VAD...")
        
        # Load Silero VAD model
        model = load_silero_vad()
        
        # Silero VAD expects 16kHz mono audio
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            return_seconds=True,
            sampling_rate=sample_rate
        )
        
        if not speech_timestamps:
            logger.warning("No speech segments detected")
            return audio
        
        # Extract speech segments
        speech_samples = []
        for segment in speech_timestamps:
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)
            speech_samples.append(audio[start_sample:end_sample])
        
        if not speech_samples:
            logger.warning("No speech samples extracted")
            return audio
        
        # Concatenate speech segments
        result_audio = np.concatenate(speech_samples)
        
        original_duration = len(audio) / sample_rate
        new_duration = len(result_audio) / sample_rate
        reduction_percentage = ((original_duration - new_duration) / original_duration) * 100
        
        logger.info(f"Silero VAD: {original_duration:.2f}s -> {new_duration:.2f}s ({reduction_percentage:.1f}% reduction)")
        return result_audio
        
    except ImportError:
        logger.warning("Silero VAD not available, returning original audio")
        return audio
    except Exception as e:
        logger.error(f"Silero VAD failed: {e}")
        return audio


def apply_denoising(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict]:
    """Apply DeepFilterNet denoising (CPU default)"""
    try:
        from deepfilternet import dfnet
        
        logger.info("Applying DeepFilterNet denoising (CPU)...")
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Initialize DeepFilterNet model
        model = dfnet.DFNet()
        model.eval()
        
        # Process audio
        with torch.no_grad():
            denoised_audio = model(audio_tensor.unsqueeze(0)).squeeze(0).numpy()
        
        # Calculate metrics
        original_rms = np.sqrt(np.mean(audio ** 2))
        denoised_rms = np.sqrt(np.mean(denoised_audio ** 2))
        snr_improvement = 20 * np.log10(denoised_rms / (original_rms + 1e-10))
        
        metrics = {
            "applied": True,
            "method": "DeepFilterNet",
            "device": "cpu",
            "snr_improvement_db": snr_improvement
        }
        
        logger.info(f"Denoising completed: SNR improvement: {snr_improvement:.2f} dB")
        return denoised_audio, metrics
        
    except ImportError:
        logger.warning("DeepFilterNet not available")
        return audio, {"applied": False, "reason": "DeepFilterNet not available"}
    except Exception as e:
        logger.error(f"Denoising failed: {e}")
        return audio, {"applied": False, "error": str(e)}


def preprocess_audio(file_path: str, enable_denoising: bool = True) -> Dict:
    """Preprocess audio file with denoising and silence removal"""
    logger.info(f"Preprocessing audio: {file_path}")
    
    try:
        # Load audio
        audio, sample_rate = librosa.load(file_path, sr=None)
        logger.info(f"Loaded audio: {len(audio)/sample_rate:.2f}s at {sample_rate}Hz")
        
        # Apply denoising if enabled
        if enable_denoising:
            audio, denoising_metrics = apply_denoising(audio, sample_rate)
        else:
            denoising_metrics = {"applied": False, "reason": "Disabled"}
        
        # Remove long silences using Silero VAD
        if settings.remove_long_silences:
            audio = remove_long_silences_silero(audio, sample_rate)
            silence_removal_metrics = {"applied": True, "method": "Silero VAD"}
        else:
            silence_removal_metrics = {"applied": False, "reason": "Disabled"}
        
        return {
            "audio": audio,
            "sample_rate": sample_rate,
            "duration": len(audio) / sample_rate,
            "denoising_metrics": denoising_metrics,
            "silence_removal_metrics": silence_removal_metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to preprocess audio {file_path}: {e}")
        raise
