"""
DeepFilterNet-based audio denoising for TTS datasets
CPU-optimized implementation
"""
import logging
from typing import Optional, Tuple, Dict
import numpy as np

logger = logging.getLogger(__name__)

# Check if DeepFilterNet is available
try:
    import torch
    import torchaudio
    from deepfilternet import dfnet
    DEEPFILTERNET_AVAILABLE = True
except ImportError:
    DEEPFILTERNET_AVAILABLE = False


class AudioDenoiser:
    """DeepFilterNet-based audio denoising for TTS datasets."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the audio denoiser.
        
        Args:
            model_path: Path to custom DeepFilterNet model (optional)
        """
        self.model_path = model_path
        self.model = None
        self.device = "cpu"  # Default to CPU

        if not DEEPFILTERNET_AVAILABLE:
            logger.warning("DeepFilterNet not available. Install with: pip install deepfilternet")
            return

        # Set device if PyTorch is available
        if DEEPFILTERNET_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Initialize DeepFilterNet model
            self.model = dfnet.DFNet()
            self.model.eval()
            
            # Move to device
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            logger.info(f"DeepFilterNet denoiser initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepFilterNet: {e}")
            self.model = None

    def denoise(self, audio: np.ndarray, sample_rate: int = 48000) -> Tuple[np.ndarray, Dict]:
        """
        Denoise audio using DeepFilterNet.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (denoised_audio, metrics_dict)
        """
        if not self.model:
            logger.warning("DeepFilterNet model not available, returning original audio")
            return audio, {"applied": False, "reason": "Model not available"}

        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Add batch dimension if needed
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Move to device
            if self.device == "cuda":
                audio_tensor = audio_tensor.cuda()
            
            # Denoise
            with torch.no_grad():
                denoised_tensor = self.model(audio_tensor)
            
            # Convert back to numpy
            denoised_audio = denoised_tensor.squeeze(0).cpu().numpy()
            
            # Calculate metrics
            original_rms = np.sqrt(np.mean(audio ** 2))
            denoised_rms = np.sqrt(np.mean(denoised_audio ** 2))
            snr_improvement = 20 * np.log10(denoised_rms / (original_rms + 1e-10))
            
            metrics = {
                "applied": True,
                "method": "DeepFilterNet",
                "device": self.device,
                "snr_improvement_db": snr_improvement,
                "original_rms": original_rms,
                "denoised_rms": denoised_rms,
                "model_path": self.model_path
            }
            
            logger.info(f"Denoising completed: SNR improvement: {snr_improvement:.2f} dB")
            return denoised_audio, metrics
            
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            return audio, {"applied": False, "error": str(e)}

    def is_available(self) -> bool:
        """Check if denoising is available."""
        return DEEPFILTERNET_AVAILABLE and self.model is not None


def apply_denoising(audio: np.ndarray, sample_rate: int = 48000) -> Tuple[np.ndarray, Dict]:
    """
    Apply DeepFilterNet denoising to audio.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate of the audio
        
    Returns:
        Tuple of (denoised_audio, metrics_dict)
    """
    denoiser = AudioDenoiser()
    return denoiser.denoise(audio, sample_rate)