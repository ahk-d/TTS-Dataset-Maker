#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Audio Denoiser for TTS Dataset Maker

This module provides robust audio denoising capabilities with multiple algorithms
and comprehensive error handling for TTS dataset preprocessing.
"""

import os
import json
import logging
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import torch
import soundfile as sf
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Denoiser:
    """
    Professional audio denoiser with multiple algorithms and robust error handling.
    
    Supports:
    - DeepFilterNet (if available)
    - Spectral subtraction
    - Wiener filtering
    - Basic noise reduction
    """
    
    def __init__(self, method: str = "auto", model_path: Optional[str] = None):
        """
        Initialize the denoiser.
        
        Args:
            method: Denoising method ('auto', 'deepfilter', 'spectral', 'wiener', 'basic')
            model_path: Path to pre-trained model (for deep learning methods)
        """
        self.method = method
        self.model_path = model_path
        self.model = None
        self.df_state = None
        self._init_model()
    
    def _init_model(self):
        """Initialize the denoising model based on method."""
        if self.method in ["auto", "deepfilter"]:
            try:
                from df import enhance, init_df
                self.model, self.df_state, _ = init_df()
                logger.info("✓ DeepFilterNet model loaded successfully")
                self.method = "deepfilter"
            except ImportError:
                logger.warning("DeepFilterNet not available, falling back to spectral subtraction")
                self.method = "spectral"
            except Exception as e:
                logger.error(f"Failed to load DeepFilterNet: {e}")
                self.method = "spectral"
    
    def _load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with robust error handling."""
        try:
            # Try pydub first for better format support
            audio = AudioSegment.from_file(file_path)
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            return samples, audio.frame_rate
        except Exception as e:
            logger.warning(f"Pydub failed, trying soundfile: {e}")
            try:
                data, sr = sf.read(file_path, always_2d=False)
                return data, sr
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio file {file_path}: {e2}")
    
    def _save_audio(self, audio: np.ndarray, sample_rate: int, output_path: str):
        """Save audio file with robust error handling."""
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to appropriate format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize if needed
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            sf.write(output_path, audio, sample_rate)
            logger.info(f"✓ Audio saved to: {output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save audio to {output_path}: {e}")
    
    def _ensure_mono_float32(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to mono float32 format."""
        if audio.ndim == 1:
            audio = audio.astype(np.float32)
        elif audio.ndim == 2:
            audio = audio.mean(axis=1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio dimensions: {audio.ndim}")
        
        # Handle NaN and infinite values
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return audio
    
    def _spectral_subtraction(self, audio: np.ndarray, sample_rate: int, noise_factor: float = 0.1) -> np.ndarray:
        """Apply spectral subtraction denoising."""
        try:
            from scipy import signal
            from scipy.fft import fft, ifft
            
            # Convert to frequency domain
            fft_audio = fft(audio)
            
            # Estimate noise spectrum from first 10% of audio
            noise_samples = int(0.1 * len(audio))
            noise_spectrum = np.mean(np.abs(fft_audio[:noise_samples]))
            
            # Apply spectral subtraction
            denoised_fft = fft_audio - noise_factor * noise_spectrum
            denoised_fft = np.maximum(denoised_fft, 0.01 * np.abs(fft_audio))  # Floor
            
            # Convert back to time domain
            denoised_audio = np.real(ifft(denoised_fft))
            
            return denoised_audio
            
        except ImportError:
            logger.warning("Scipy not available, using basic noise reduction")
            return self._basic_noise_reduction(audio)
    
    def _wiener_filter(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply Wiener filtering for denoising."""
        try:
            from scipy import signal
            
            # Estimate noise from first 10% of audio
            noise_samples = int(0.1 * len(audio))
            noise_estimate = audio[:noise_samples]
            
            # Apply Wiener filter
            denoised_audio = signal.wiener(audio, mysize=len(noise_estimate))
            
            return denoised_audio
            
        except ImportError:
            logger.warning("Scipy not available, using basic noise reduction")
            return self._basic_noise_reduction(audio)
    
    def _basic_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Basic noise reduction using simple filtering."""
        # Simple high-pass filter to remove low-frequency noise
        # This is a very basic implementation
        denoised = audio.copy()
        
        # Remove DC offset
        denoised = denoised - np.mean(denoised)
        
        # Simple smoothing (very basic)
        window_size = 3
        if len(denoised) > window_size:
            denoised = np.convolve(denoised, np.ones(window_size)/window_size, mode='same')
        
        return denoised
    
    def _deepfilter_denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply DeepFilterNet denoising."""
        try:
            from df import enhance
            
            # Prepare audio for DeepFilterNet
            audio_mono = self._ensure_mono_float32(audio)
            audio_tensor = torch.from_numpy(audio_mono)
            
            # Apply denoising
            with torch.no_grad():
                enhanced = enhance(self.model, self.df_state, audio_tensor)
            
            # Convert back to numpy
            denoised_audio = enhanced.squeeze().cpu().numpy()
            
            return denoised_audio
            
        except Exception as e:
            logger.error(f"DeepFilterNet denoising failed: {e}")
            return self._spectral_subtraction(audio, sample_rate)
    
    def denoise_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Denoise audio using the selected method."""
        logger.info(f"Applying {self.method} denoising...")
        
        # Ensure audio is in correct format
        audio = self._ensure_mono_float32(audio)
        
        # Apply denoising based on method
        if self.method == "deepfilter":
            return self._deepfilter_denoise(audio, sample_rate)
        elif self.method == "spectral":
            return self._spectral_subtraction(audio, sample_rate)
        elif self.method == "wiener":
            return self._wiener_filter(audio, sample_rate)
        elif self.method == "basic":
            return self._basic_noise_reduction(audio)
        else:
            raise ValueError(f"Unknown denoising method: {self.method}")
    
    def denoise_file(self, input_path: str, output_path: str) -> bool:
        """
        Denoise an audio file and save the result.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save denoised audio
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Denoising: {input_path}")
            
            # Load audio
            audio, sample_rate = self._load_audio(input_path)
            
            # Apply denoising
            denoised_audio = self.denoise_audio(audio, sample_rate)
            
            # Save result
            self._save_audio(denoised_audio, sample_rate, output_path)
            
            logger.info(f"✓ Denoising completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            return False
    
    def denoise_segments(self, metadata_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Denoise all segments in a metadata file.
        
        Args:
            metadata_path: Path to metadata.json file
            output_dir: Directory to save denoised segments
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            segments = metadata.get("segments", [])
            if not segments:
                raise ValueError("No segments found in metadata")
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Process each segment
            successful = 0
            failed = 0
            denoised_segments = []
            
            for i, segment in enumerate(segments):
                try:
                    # Get input path
                    input_path = segment.get("audio_path")
                    if not input_path or not os.path.exists(input_path):
                        logger.warning(f"Segment {i}: Audio file not found: {input_path}")
                        failed += 1
                        continue
                    
                    # Create output path
                    segment_id = segment.get("id", i)
                    output_filename = f"segment_{segment_id:06d}_denoised.wav"
                    output_filepath = output_path / output_filename
                    
                    # Denoise segment
                    if self.denoise_file(input_path, str(output_filepath)):
                        # Update segment metadata
                        denoised_segment = segment.copy()
                        denoised_segment["audio_path"] = str(output_filepath)
                        denoised_segment["denoised"] = True
                        denoised_segment["denoising_method"] = self.method
                        denoised_segments.append(denoised_segment)
                        successful += 1
                        
                        logger.info(f"✓ Segment {i+1}/{len(segments)}: {segment_id}")
                    else:
                        failed += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process segment {i}: {e}")
                    failed += 1
            
            # Create denoised metadata
            denoised_metadata = {
                "dataset_info": {
                    "original_metadata": metadata_path,
                    "denoising_method": self.method,
                    "total_segments": len(segments),
                    "successful_denoising": successful,
                    "failed_denoising": failed,
                    "denoised_at": pd.Timestamp.now().isoformat()
                },
                "segments": denoised_segments
            }
            
            # Save denoised metadata
            denoised_metadata_path = output_path / "denoised_metadata.json"
            with open(denoised_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(denoised_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Denoising completed: {successful} successful, {failed} failed")
            logger.info(f"✓ Denoised metadata saved to: {denoised_metadata_path}")
            
            return {
                "successful": successful,
                "failed": failed,
                "total": len(segments),
                "denoised_metadata_path": str(denoised_metadata_path),
                "output_directory": str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Segment denoising failed: {e}")
            return {"successful": 0, "failed": 0, "total": 0, "error": str(e)}

def main():
    """Command-line interface for the denoiser."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Professional Audio Denoiser")
    parser.add_argument("input", help="Input audio file or metadata.json")
    parser.add_argument("output", help="Output file or directory")
    parser.add_argument("--method", choices=["auto", "deepfilter", "spectral", "wiener", "basic"], 
                       default="auto", help="Denoising method")
    parser.add_argument("--segments", action="store_true", help="Process metadata segments")
    
    args = parser.parse_args()
    
    # Initialize denoiser
    denoiser = Denoiser(method=args.method)
    
    if args.segments:
        # Process segments
        result = denoiser.denoise_segments(args.input, args.output)
        print(f"Results: {result}")
    else:
        # Process single file
        success = denoiser.denoise_file(args.input, args.output)
        print(f"Success: {success}")

if __name__ == "__main__":
    main()
