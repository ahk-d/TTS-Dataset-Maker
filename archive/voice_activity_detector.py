"""
Voice Activity Detection (VAD) module for TTS Dataset Maker
Detects speech activity and filters out silent segments
"""
import os
import logging
from typing import List, Tuple, Optional
import numpy as np
from pydub import AudioSegment
import webrtcvad

from config import settings


logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Handles voice activity detection and silent segment filtering"""
    
    def __init__(self, method: str = None, min_speech_duration: float = None):
        """
        Initialize voice activity detector
        
        Args:
            method: VAD method (defaults to settings)
            min_speech_duration: Minimum speech duration in seconds (defaults to settings)
        """
        self.method = method or settings.vad_method
        self.min_speech_duration = min_speech_duration or settings.min_speech_duration
        self.silence_threshold = settings.silence_threshold
        self.min_speech_ratio = settings.min_speech_ratio
        
        # Initialize VAD model
        self.vad = None
        if self.method == "webrtcvad":
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
        
        logger.info(f"VAD initialized with method: {self.method}, min_speech_duration: {self.min_speech_duration}s")
    
    def detect_speech_activity(self, audio: AudioSegment) -> List[Tuple[float, float]]:
        """
        Detect speech activity in audio segment
        
        Args:
            audio: AudioSegment to analyze
            
        Returns:
            List of (start_time, end_time) tuples for speech segments
        """
        try:
            if self.method == "webrtcvad":
                return self._detect_with_webrtcvad(audio)
            elif self.method == "librosa":
                return self._detect_with_librosa(audio)
            else:
                raise ValueError(f"Unsupported VAD method: {self.method}")
                
        except Exception as e:
            logger.error(f"VAD detection failed: {e}")
            # Return full audio as speech if detection fails
            return [(0.0, len(audio) / 1000.0)]
    
    def _detect_with_webrtcvad(self, audio: AudioSegment) -> List[Tuple[float, float]]:
        """
        Detect speech using WebRTC VAD
        
        Args:
            audio: AudioSegment to analyze
            
        Returns:
            List of speech segments
        """
        try:
            # WebRTC VAD requires 8kHz, 16kHz, 22kHz, or 32kHz mono audio
            # Convert to 16kHz mono if needed
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Convert to raw audio data
            raw_audio = audio.raw_data
            
            # WebRTC VAD works on 10ms, 20ms, or 30ms frames
            frame_duration = 20  # ms
            frame_size = int(16000 * frame_duration / 1000)  # samples per frame
            frame_bytes = frame_size * 2  # 16-bit audio
            
            speech_segments = []
            current_speech_start = None
            
            # Process audio in frames
            for i in range(0, len(raw_audio), frame_bytes):
                frame = raw_audio[i:i + frame_bytes]
                
                # Pad frame if it's too short
                if len(frame) < frame_bytes:
                    frame += b'\x00' * (frame_bytes - len(frame))
                
                # Check if frame contains speech
                is_speech = self.vad.is_speech(frame, 16000)
                frame_time = i / (16000 * 2)  # Convert to seconds
                
                if is_speech:
                    if current_speech_start is None:
                        current_speech_start = frame_time
                else:
                    if current_speech_start is not None:
                        # End of speech segment
                        speech_duration = frame_time - current_speech_start
                        if speech_duration >= self.min_speech_duration:
                            speech_segments.append((current_speech_start, frame_time))
                        current_speech_start = None
            
            # Handle case where speech continues to end of audio
            if current_speech_start is not None:
                end_time = len(audio) / 1000.0
                speech_duration = end_time - current_speech_start
                if speech_duration >= self.min_speech_duration:
                    speech_segments.append((current_speech_start, end_time))
            
            logger.info(f"WebRTC VAD detected {len(speech_segments)} speech segments")
            return speech_segments
            
        except Exception as e:
            logger.error(f"WebRTC VAD detection failed: {e}")
            return [(0.0, len(audio) / 1000.0)]
    
    def _detect_with_librosa(self, audio: AudioSegment) -> List[Tuple[float, float]]:
        """
        Detect speech using librosa-based energy and spectral analysis
        
        Args:
            audio: AudioSegment to analyze
            
        Returns:
            List of speech segments
        """
        try:
            import librosa
            
            # Convert to numpy array
            audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
            sample_rate = audio.frame_rate
            
            # Normalize audio
            if audio_array.max() > 0:
                audio_array = audio_array / audio_array.max()
            
            # Calculate energy
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Convert to dB
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            
            # Detect speech frames (above silence threshold)
            speech_frames = rms_db > self.silence_threshold
            
            # Convert frame indices to time
            times = librosa.frames_to_time(np.arange(len(speech_frames)), sr=sample_rate, hop_length=hop_length)
            
            # Find speech segments
            speech_segments = []
            current_speech_start = None
            
            for i, is_speech in enumerate(speech_frames):
                frame_time = times[i]
                
                if is_speech:
                    if current_speech_start is None:
                        current_speech_start = frame_time
                else:
                    if current_speech_start is not None:
                        # End of speech segment
                        speech_duration = frame_time - current_speech_start
                        if speech_duration >= self.min_speech_duration:
                            speech_segments.append((current_speech_start, frame_time))
                        current_speech_start = None
            
            # Handle case where speech continues to end
            if current_speech_start is not None:
                end_time = len(audio_array) / sample_rate
                speech_duration = end_time - current_speech_start
                if speech_duration >= self.min_speech_duration:
                    speech_segments.append((current_speech_start, end_time))
            
            logger.info(f"Librosa VAD detected {len(speech_segments)} speech segments")
            return speech_segments
            
        except ImportError:
            logger.error("Librosa not available, falling back to energy-based detection")
            return self._detect_with_energy(audio)
        except Exception as e:
            logger.error(f"Librosa VAD detection failed: {e}")
            return self._detect_with_energy(audio)
    
    def _detect_with_energy(self, audio: AudioSegment) -> List[Tuple[float, float]]:
        """
        Simple energy-based speech detection fallback
        
        Args:
            audio: AudioSegment to analyze
            
        Returns:
            List of speech segments
        """
        try:
            # Convert to numpy array
            audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
            sample_rate = audio.frame_rate
            
            # Normalize
            if audio_array.max() > 0:
                audio_array = audio_array / audio_array.max()
            
            # Calculate energy in frames
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            
            speech_segments = []
            current_speech_start = None
            
            for i in range(0, len(audio_array) - frame_length, hop_length):
                frame = audio_array[i:i + frame_length]
                energy = np.mean(frame ** 2)
                energy_db = 10 * np.log10(energy + 1e-10)  # Add small value to avoid log(0)
                
                frame_time = i / sample_rate
                is_speech = energy_db > self.silence_threshold
                
                if is_speech:
                    if current_speech_start is None:
                        current_speech_start = frame_time
                else:
                    if current_speech_start is not None:
                        # End of speech segment
                        speech_duration = frame_time - current_speech_start
                        if speech_duration >= self.min_speech_duration:
                            speech_segments.append((current_speech_start, frame_time))
                        current_speech_start = None
            
            # Handle case where speech continues to end
            if current_speech_start is not None:
                end_time = len(audio_array) / sample_rate
                speech_duration = end_time - current_speech_start
                if speech_duration >= self.min_speech_duration:
                    speech_segments.append((current_speech_start, end_time))
            
            logger.info(f"Energy-based VAD detected {len(speech_segments)} speech segments")
            return speech_segments
            
        except Exception as e:
            logger.error(f"Energy-based VAD detection failed: {e}")
            return [(0.0, len(audio) / 1000.0)]
    
    def filter_silent_segments(self, segments: List, audio: AudioSegment) -> List:
        """
        Filter out segments that don't contain speech
        
        Args:
            segments: List of segment objects with start_time and end_time
            audio: Original audio segment
            
        Returns:
            List of segments that contain speech
        """
        try:
            logger.info(f"Filtering {len(segments)} segments for speech activity")
            
            # Detect speech activity in the full audio
            speech_segments = self.detect_speech_activity(audio)
            
            # Filter segments that overlap with speech
            speech_containing_segments = []
            
            for segment in segments:
                segment_start = segment.start_time
                segment_end = segment.end_time
                
                # Check if segment overlaps with any speech segment
                has_speech = False
                speech_ratio = 0.0
                
                for speech_start, speech_end in speech_segments:
                    # Calculate overlap
                    overlap_start = max(segment_start, speech_start)
                    overlap_end = min(segment_end, speech_end)
                    
                    if overlap_start < overlap_end:
                        overlap_duration = overlap_end - overlap_start
                        segment_duration = segment_end - segment_start
                        speech_ratio = overlap_duration / segment_duration
                        
                        if speech_ratio >= self.min_speech_ratio:
                            has_speech = True
                            break
                
                if has_speech:
                    # Add speech activity info to segment
                    segment.speech_ratio = speech_ratio
                    segment.has_speech = True
                    speech_containing_segments.append(segment)
                else:
                    logger.debug(f"Filtered out silent segment: {segment.segment_id}")
            
            logger.info(f"Filtered to {len(speech_containing_segments)} speech-containing segments")
            return speech_containing_segments
            
        except Exception as e:
            logger.error(f"Failed to filter silent segments: {e}")
            return segments
    
    def get_vad_info(self) -> dict:
        """
        Get information about VAD configuration
        
        Returns:
            Dictionary with VAD info
        """
        return {
            "method": self.method,
            "min_speech_duration": self.min_speech_duration,
            "silence_threshold": self.silence_threshold,
            "min_speech_ratio": self.min_speech_ratio,
            "vad_available": self.vad is not None
        }
