#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Audio Denoiser using DeepFilterNet

This module provides high-quality audio denoising using DeepFilterNet,
a state-of-the-art deep learning-based noise suppression system.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import soundfile as sf
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Denoiser:
    """
    Professional audio denoiser using DeepFilterNet.
    
    DeepFilterNet provides state-of-the-art noise suppression for speech audio,
    making it ideal for TTS dataset preprocessing.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the DeepFilterNet denoiser.
        
        Args:
            model_path: Path to pre-trained DeepFilterNet model (optional)
        """
        self.model = None
        self.df_state = None
        self.model_path = model_path
        self._init_deepfilter()
    
    def _init_deepfilter(self):
        """Initialize DeepFilterNet model."""
        try:
            from df import enhance, init_df
            
            # Initialize DeepFilterNet
            if self.model_path and os.path.exists(self.model_path):
                self.model, self.df_state, _ = init_df(model_path=self.model_path)
                logger.info(f"✓ DeepFilterNet model loaded from: {self.model_path}")
            else:
                self.model, self.df_state, _ = init_df()
                logger.info("✓ DeepFilterNet model loaded (default)")
                
        except ImportError:
            raise ImportError(
                "DeepFilterNet not installed. Install with: "
                "pip install deepfilternet"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DeepFilterNet: {e}")
    
    def _load_audio(self, file_path: str) -> tuple[np.ndarray, int]:
        """Load audio file with robust error handling."""
        try:
            # Try soundfile first for better format support
            data, sr = sf.read(file_path, always_2d=False)
            logger.info(f"✓ Loaded audio: {file_path} ({sr} Hz)")
            return data, sr
        except Exception as e:
            logger.warning(f"Soundfile failed, trying pydub: {e}")
            try:
                audio = AudioSegment.from_file(file_path)
                samples = np.array(audio.get_array_of_samples())
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2))
                logger.info(f"✓ Loaded audio with pydub: {file_path} ({audio.frame_rate} Hz)")
                return samples, audio.frame_rate
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
    
    def _prepare_audio_for_deepfilter(self, audio: np.ndarray) -> torch.Tensor:
        """Prepare audio for DeepFilterNet processing."""
        # Ensure mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        # Convert to float32
        audio = audio.astype(np.float32)
        
        # Handle NaN and infinite values
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure it's 1D
        if audio.ndim != 1:
            audio = audio.flatten()
        
        # Convert to torch tensor and ensure correct shape
        audio_tensor = torch.from_numpy(audio)
        
        # DeepFilterNet expects (batch_size, channels, samples) or (channels, samples)
        # We have (samples), so we need to add channel dimension
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
        
        return audio_tensor
    
    def denoise_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Denoise audio using DeepFilterNet.
        
        Args:
            audio: Input audio as numpy array
            sample_rate: Audio sample rate
            
        Returns:
            Denoised audio as numpy array
        """
        try:
            from df import enhance
            
            logger.info("Applying DeepFilterNet denoising...")
            
            # Prepare audio for DeepFilterNet
            audio_tensor = self._prepare_audio_for_deepfilter(audio)
            
            # Apply DeepFilterNet denoising
            with torch.no_grad():
                enhanced = enhance(self.model, self.df_state, audio_tensor)
            
            # Convert back to numpy
            denoised_audio = enhanced.squeeze().cpu().numpy()
            
            logger.info("✓ DeepFilterNet denoising completed")
            return denoised_audio
            
        except Exception as e:
            logger.error(f"DeepFilterNet denoising failed: {e}")
            raise RuntimeError(f"DeepFilterNet denoising failed: {e}")
    
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
            logger.info(f"Denoising file: {input_path}")
            
            # Load audio
            audio, sample_rate = self._load_audio(input_path)
            
            # Apply DeepFilterNet denoising
            denoised_audio = self.denoise_audio(audio, sample_rate)
            
            # Save result
            self._save_audio(denoised_audio, sample_rate, output_path)
            
            logger.info(f"✓ File denoising completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"File denoising failed: {e}")
            # Don't create an empty or corrupted file
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    
    def denoise_segments(self, metadata_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Denoise all segments in a metadata file using DeepFilterNet.
        
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
            
            logger.info(f"Processing {len(segments)} segments with DeepFilterNet...")
            
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
                    
                    # Denoise segment with DeepFilterNet
                    if self.denoise_file(input_path, str(output_filepath)):
                        # Update segment metadata
                        denoised_segment = segment.copy()
                        denoised_segment["audio_path"] = str(output_filepath)
                        denoised_segment["denoised"] = True
                        denoised_segment["denoising_method"] = "deepfilternet"
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
                    "denoising_method": "deepfilternet",
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
            
            logger.info(f"✓ DeepFilterNet denoising completed: {successful} successful, {failed} failed")
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
    """Command-line interface for the DeepFilterNet denoiser."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Professional Audio Denoiser using DeepFilterNet")
    parser.add_argument("input", help="Input audio file or metadata.json")
    parser.add_argument("output", help="Output file or directory")
    parser.add_argument("--model-path", help="Path to DeepFilterNet model")
    parser.add_argument("--segments", action="store_true", help="Process metadata segments")
    
    args = parser.parse_args()
    
    # Initialize DeepFilterNet denoiser
    denoiser = Denoiser(model_path=args.model_path)
    
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
