"""
Audio processing tasks for TTS dataset creation
Includes Silero VAD-based silence removal and denoising
"""
import gc
import logging
from functools import lru_cache
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch

from df import config as df_config
from df.enhance import enhance, init_df
from df.io import resample

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

@lru_cache(maxsize=1)
def _init_deepfilternet():
    """Initialise DeepFilterNet (df) model/state once."""
    logger.info("Initialising DeepFilterNet model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("DeepFilterNet device: %s", device)

    model, df_state, _ = init_df(config_allow_defaults=True)
    model = model.to(device=device).eval()

    return model, df_state, device


def denoise_with_deepfilternet(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict]:
    """Denoise audio using the modern DeepFilterNet (df) API with chunking."""
    
    # Temporary emergency bypass for CUDA issues
    import os
    if os.environ.get("DISABLE_DEEPFILTERNET", "false").lower() == "true":
        logger.warning("DeepFilterNet disabled via DISABLE_DEEPFILTERNET environment variable")
        return audio, {"applied": False, "reason": "Disabled via environment variable"}

    try:
        model, df_state, device = _init_deepfilternet()
    except Exception as exc:
        logger.warning("DeepFilterNet initialisation failed: %s", exc)
        return audio, {"applied": False, "reason": str(exc)}
    
    logger.info("Starting DeepFilterNet denoising process")

    try:
        logger.info("Applying DeepFilterNet denoising")
        target_sr = settings.denoising_sample_rate

        logger.debug("Step 1: Converting audio to tensor")
        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != target_sr:
            logger.info("Resampling %d Hz -> %d Hz", sample_rate, target_sr)
            try:
                logger.debug("Step 2a: Using df.io.resample")
                # Try df.io.resample first
                audio_tensor = resample(audio_tensor.unsqueeze(0), sample_rate, target_sr).squeeze(0).detach().cpu()
            except Exception as e:
                logger.warning("df.io.resample failed, falling back to librosa: %s", e)
                logger.debug("Step 2b: Using librosa fallback")
                # Fallback to librosa resampling
                audio_np = audio_tensor.detach().cpu().numpy()
                audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=target_sr)
                audio_tensor = torch.from_numpy(audio_np).float()
            sr = target_sr
        else:
            sr = sample_rate

        logger.debug("Step 3: Adding batch dimension")
        audio_tensor = audio_tensor.unsqueeze(0)

        chunk_samples = int(settings.denoising_chunk_duration * sr)
        overlap_samples = int(settings.denoising_overlap_duration * sr)

        if audio_tensor.shape[-1] <= chunk_samples:
            logger.debug("Step 4a: Processing short audio (single chunk)")
            with torch.no_grad():
                enhanced = enhance(model, df_state, audio_tensor.to(device=device)).detach().cpu()
        else:
            logger.info(
                "Processing in %.1fs chunks with %.1fs overlap",
                settings.denoising_chunk_duration,
                settings.denoising_overlap_duration,
            )
            logger.debug("Step 4b: Processing long audio (chunked)")

            enhanced_chunks: List[torch.Tensor] = []
            num_samples = audio_tensor.shape[-1]
            step_samples = chunk_samples - overlap_samples
            num_chunks = int(np.ceil((num_samples - overlap_samples) / step_samples))

            for idx in range(num_chunks):
                logger.debug("Processing chunk %d/%d", idx + 1, num_chunks)
                start_idx = idx * step_samples
                end_idx = min(start_idx + chunk_samples, num_samples)

                chunk = audio_tensor[..., start_idx:end_idx]

                with torch.no_grad():
                    enhanced_chunk = enhance(model, df_state, chunk.to(device=device)).detach().cpu()

                if idx == 0:
                    enhanced_chunks.append(enhanced_chunk)
                else:
                    enhanced_chunks.append(enhanced_chunk[..., overlap_samples:])

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            logger.debug("Step 5: Concatenating chunks")
            enhanced = torch.cat(enhanced_chunks, dim=-1).detach().cpu()
            enhanced = enhanced[..., :num_samples]

        logger.debug("Step 6: Converting to numpy - Enhanced tensor device: %s, requires_grad: %s", enhanced.device, enhanced.requires_grad)
        enhanced_audio = enhanced.squeeze(0).numpy()

        if sample_rate != sr:
            logger.info("Resampling denoised audio back to %d Hz", sample_rate)
            enhanced_audio = librosa.resample(enhanced_audio, orig_sr=sr, target_sr=sample_rate)

        original_rms = float(np.sqrt(np.mean(audio ** 2)))
        denoised_rms = float(np.sqrt(np.mean(enhanced_audio ** 2)))
        snr_improvement = 20 * np.log10((denoised_rms + 1e-10) / (original_rms + 1e-10))

        metrics = {
            "applied": True,
            "method": "DeepFilterNet",
            "device": str(device),
            "snr_improvement_db": snr_improvement,
        }

        logger.info("DeepFilterNet denoising complete (ΔSNR %.2f dB)", snr_improvement)
        return enhanced_audio, metrics
        
    except Exception as exc:
        logger.error("DeepFilterNet processing failed at some step: %s", exc, exc_info=True)
        return audio, {"applied": False, "reason": f"Processing failed: {exc}"}


def preprocess_audio(file_path: str, enable_denoising: bool = True) -> Dict:
    """Preprocess audio file with denoising followed by optional VAD."""
    logger.info("Preprocessing audio: %s", file_path)

    try:
        audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
        logger.info("Loaded audio: %.2fs at %d Hz", len(audio) / sample_rate, sample_rate)

        if enable_denoising and settings.denoising_enabled:
            audio, denoising_metrics = denoise_with_deepfilternet(audio, sample_rate)
        else:
            denoising_metrics = {"applied": False, "reason": "Disabled"}

        if settings.remove_long_silences and settings.vad_enabled:
            audio = remove_long_silences_silero(audio, sample_rate)
            silence_removal_metrics = {"applied": True, "method": "Silero VAD"}
        else:
            silence_removal_metrics = {"applied": False, "reason": "Disabled"}

        return {
            "audio": audio,
            "sample_rate": sample_rate,
            "duration": len(audio) / sample_rate,
            "denoising_metrics": denoising_metrics,
            "silence_removal_metrics": silence_removal_metrics,
        }

    except Exception as exc:
        logger.error("Failed to preprocess audio %s: %s", file_path, exc)
        raise
