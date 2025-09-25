"""
Audio processing module for TTS Dataset Maker
Handles audio preprocessing, resampling, and quality validation
"""
import os
import logging
from typing import List, Tuple, Optional
from pathlib import Path

import librosa
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
import torchaudio

from config import settings
# AudioDenoiser removed - no denoising
from voice_activity_detector import VoiceActivityDetector


logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio preprocessing and validation"""
    
    def __init__(self):
        self.target_sample_rate = settings.default_sample_rate
        self.min_duration = settings.min_segment_duration
        self.max_duration = settings.max_segment_duration
        
        # Initialize VAD (denoising removed)
        self.vad = VoiceActivityDetector() if settings.vad_enabled else None
        
    def load_audio(self, file_path: str) -> AudioSegment:
        """
        Load audio file and return AudioSegment object
        
        Args:
            file_path: Path to audio file
            
        Returns:
            AudioSegment object
        """
        try:
            audio = AudioSegment.from_file(file_path)
            logger.info(f"Loaded audio: {file_path}, duration: {len(audio)/1000:.2f}s")
            return audio
        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            raise
    
    def resample_audio(self, audio: AudioSegment, target_rate: int = None) -> AudioSegment:
        """
        Resample audio to target sample rate
        
        Args:
            audio: AudioSegment object
            target_rate: Target sample rate (defaults to settings)
            
        Returns:
            Resampled AudioSegment
        """
        if target_rate is None:
            target_rate = self.target_sample_rate
            
        if audio.frame_rate != target_rate:
            logger.info(f"Resampling from {audio.frame_rate}Hz to {target_rate}Hz")
            audio = audio.set_frame_rate(target_rate)
            
        return audio
    
    def convert_to_mono(self, audio: AudioSegment) -> AudioSegment:
        """
        Convert audio to mono channel
        
        Args:
            audio: AudioSegment object
            
        Returns:
            Mono AudioSegment
        """
        if audio.channels > 1:
            logger.info(f"Converting from {audio.channels} channels to mono")
            audio = audio.set_channels(1)
        return audio
    
    def normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """
        Normalize audio levels
        
        Args:
            audio: AudioSegment object
            
        Returns:
            Normalized AudioSegment
        """
        try:
            normalized = normalize(audio)
            logger.info("Audio normalized")
            return normalized
        except Exception as e:
            logger.warning(f"Failed to normalize audio: {e}")
            return audio
    
    
    def validate_audio_quality(self, audio: AudioSegment) -> Tuple[bool, List[str]]:
        """
        Validate audio quality and return issues
        
        Args:
            audio: AudioSegment object
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check duration
        duration = len(audio) / 1000.0  # Convert to seconds
        if duration < self.min_duration:
            issues.append(f"Audio too short: {duration:.2f}s < {self.min_duration}s")
        elif duration > self.max_duration:
            issues.append(f"Audio too long: {duration:.2f}s > {self.max_duration}s")
        
        # Check sample rate
        if audio.frame_rate != self.target_sample_rate:
            issues.append(f"Wrong sample rate: {audio.frame_rate}Hz != {self.target_sample_rate}Hz")
        
        # Check channels
        if audio.channels != 1:
            issues.append(f"Not mono: {audio.channels} channels")
        
        # Check for silence
        if audio.max_dBFS < -60:
            issues.append("Audio appears to be silent or very quiet")
        
        # Check for clipping
        if audio.max_dBFS > -0.1:
            issues.append("Audio may be clipped")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def preprocess_audio(self, file_path: str) -> Tuple[AudioSegment, bool, List[str]]:
        """
        Complete audio preprocessing pipeline
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (processed_audio, is_valid, issues)
        """
        try:
            # Load audio
            audio = self.load_audio(file_path)
            
            # Denoising removed - processing original audio
            
            # Convert to mono
            audio = self.convert_to_mono(audio)
            
            # Resample
            audio = self.resample_audio(audio)
            
            # Normalize
            audio = self.normalize_audio(audio)
            
            # Validate
            is_valid, issues = self.validate_audio_quality(audio)
            
            if not is_valid:
                logger.warning(f"Audio quality issues for {file_path}: {issues}")
            else:
                logger.info(f"Audio preprocessing successful for {file_path}")
            
            return audio, is_valid, issues
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed for {file_path}: {e}")
            raise
    
    def segment_audio(self, audio: AudioSegment, start_time: float, end_time: float, 
                     overlap_ms: int = None) -> AudioSegment:
        """
        Extract audio segment with optional overlap
        
        Args:
            audio: AudioSegment object
            start_time: Start time in seconds
            end_time: End time in seconds
            overlap_ms: Overlap in milliseconds
            
        Returns:
            AudioSegment for the specified time range
        """
        if overlap_ms is None:
            overlap_ms = settings.audio_overlap_ms
            
        # Convert to milliseconds
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        # Add overlap
        start_ms = max(0, start_ms - overlap_ms)
        end_ms = min(len(audio), end_ms + overlap_ms)
        
        # Extract segment
        segment = audio[start_ms:end_ms]
        
        logger.debug(f"Extracted segment: {start_time:.2f}s - {end_time:.2f}s "
                    f"({len(segment)/1000:.2f}s duration)")
        
        return segment
    
    def save_audio(self, audio: AudioSegment, output_path: str, format: str = "wav") -> None:
        """
        Save audio segment to file
        
        Args:
            audio: AudioSegment object
            output_path: Output file path
            format: Audio format (default: wav)
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export audio
            audio.export(output_path, format=format)
            logger.info(f"Saved audio segment: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audio segment {output_path}: {e}")
            raise
    
    def get_audio_info(self, file_path: str) -> dict:
        """
        Get audio file information
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            audio = self.load_audio(file_path)
            
            info = {
                "file_path": file_path,
                "duration": len(audio) / 1000.0,
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
                "bit_depth": audio.sample_width * 8,
                "max_dBFS": audio.max_dBFS,
                "rms": audio.rms
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get audio info for {file_path}: {e}")
            raise
