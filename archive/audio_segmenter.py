"""
Audio segmentation module for TTS Dataset Maker
Handles segmentation of audio files based on speaker diarization results
"""
import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import uuid

from pydantic import BaseModel

from audio_processor import AudioProcessor
from assemblyai_client import SpeakerSegment, TranscriptionResult
from voice_activity_detector import VoiceActivityDetector
from config import settings


logger = logging.getLogger(__name__)


class AudioSegmentData(BaseModel):
    """Model for audio segment data"""
    segment_id: str
    start_time: float
    end_time: float
    speaker: str
    text: str
    audio_file: str
    source: str
    duration: float
    confidence: float
    quality_score: Optional[float] = None


class SegmentationResult(BaseModel):
    """Model for segmentation result"""
    segments: List[AudioSegmentData]
    total_segments: int
    total_duration: float
    speakers: Dict[str, Dict]
    processing_info: Dict


class AudioSegmenter:
    """Handles audio segmentation based on speaker diarization"""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize audio segmenter
        
        Args:
            output_dir: Output directory for segmented audio files
        """
        self.output_dir = output_dir or settings.output_dir
        self.audio_processor = AudioProcessor()
        self.vad = VoiceActivityDetector() if settings.vad_enabled else None
        
        # Create output directories
        self.segments_dir = os.path.join(self.output_dir, "audio_segments")
        self.metadata_dir = os.path.join(self.output_dir, "metadata")
        self.original_dir = os.path.join(self.output_dir, "original_files")
        
        os.makedirs(self.segments_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.original_dir, exist_ok=True)
        
        logger.info(f"Audio segmenter initialized with output dir: {self.output_dir}")
    
    def filter_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Filter segments based on quality criteria
        
        Args:
            segments: List of speaker segments
            
        Returns:
            Filtered list of segments
        """
        filtered_segments = []
        
        for segment in segments:
            # Check duration
            duration = segment.end - segment.start
            if duration < settings.min_segment_duration:
                logger.debug(f"Skipping short segment: {duration:.2f}s")
                continue
            
            if duration > settings.max_segment_duration:
                logger.debug(f"Skipping long segment: {duration:.2f}s")
                continue
            
            # Check confidence
            if segment.confidence < settings.min_confidence_score:
                logger.debug(f"Skipping low confidence segment: {segment.confidence:.2f}")
                continue
            
            # Check text content
            if not segment.text.strip():
                logger.debug("Skipping empty segment")
                continue
            
            # Check for minimum text length
            if len(segment.text.strip()) < 10:
                logger.debug(f"Skipping segment with too little text: {len(segment.text)} chars")
                continue
            
            filtered_segments.append(segment)
        
        logger.info(f"Filtered {len(segments)} segments to {len(filtered_segments)} valid segments")
        return filtered_segments
    
    def create_segments_directory(self) -> str:
        """
        Create directory for all audio segments (flat structure)
        
        Returns:
            Path to segments directory
        """
        os.makedirs(self.segments_dir, exist_ok=True)
        return self.segments_dir
    
    def generate_segment_filename(self, speaker_id: str, segment_index: int, 
                                 source_file: str) -> str:
        """
        Generate filename for audio segment
        
        Args:
            speaker_id: Speaker identifier
            segment_index: Index of segment
            source_file: Original source file name
            
        Returns:
            Generated filename
        """
        source_name = Path(source_file).stem
        return f"{source_name}_{speaker_id}_segment_{segment_index:03d}.wav"
    
    def calculate_quality_score(self, segment: SpeakerSegment, 
                              audio_segment) -> float:
        """
        Calculate quality score for segment
        
        Args:
            segment: Speaker segment data
            audio_segment: AudioSegment object
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        score = 1.0
        
        # Confidence factor
        score *= segment.confidence
        
        # Duration factor (prefer segments between 2-10 seconds)
        duration = segment.end - segment.start
        if duration < 2.0:
            score *= 0.8
        elif duration > 10.0:
            score *= 0.9
        
        # Audio quality factor
        if audio_segment.max_dBFS < -40:
            score *= 0.7  # Too quiet
        elif audio_segment.max_dBFS > -1:
            score *= 0.8  # Possible clipping
        
        # Text length factor
        text_length = len(segment.text.strip())
        if text_length < 20:
            score *= 0.9
        elif text_length > 200:
            score *= 0.95
        
        return min(1.0, max(0.0, score))
    
    def segment_audio_file(self, audio_file: str, transcription_result: TranscriptionResult,
                          copy_original: bool = True) -> SegmentationResult:
        """
        Segment audio file based on transcription results
        
        Args:
            audio_file: Path to audio file
            transcription_result: Transcription result from AssemblyAI
            copy_original: Whether to copy original file to output directory
            
        Returns:
            SegmentationResult object
        """
        try:
            logger.info(f"Segmenting audio file: {audio_file}")
            
            # Load audio (skip preprocessing since it was already done in the main pipeline)
            audio = self.audio_processor.load_audio(audio_file)
            # Just validate without reprocessing
            is_valid, issues = self.audio_processor.validate_audio_quality(audio)
            
            if not is_valid:
                logger.warning(f"Audio quality issues: {issues}")
            
            # Filter segments
            filtered_segments = self.filter_segments(transcription_result.segments)
            
            if not filtered_segments:
                logger.warning("No valid segments found after filtering")
                return SegmentationResult(
                    segments=[],
                    total_segments=0,
                    total_duration=0.0,
                    speakers={},
                    processing_info={"error": "No valid segments found"}
                )
            
            # Apply VAD filtering to remove silent segments
            if self.vad is not None:
                logger.info("Applying VAD filtering to remove silent segments")
                original_count = len(filtered_segments)
                filtered_segments = self.vad.filter_silent_segments(filtered_segments, audio)
                logger.info(f"VAD filtering: {original_count} -> {len(filtered_segments)} segments")
                
                if not filtered_segments:
                    logger.warning("No speech-containing segments found after VAD filtering")
                    return SegmentationResult(
                        segments=[],
                        total_segments=0,
                        total_duration=0.0,
                        speakers={},
                        processing_info={"error": "No speech-containing segments found"}
                    )
            
            # Copy original file if requested
            if copy_original:
                original_filename = os.path.basename(audio_file)
                original_dest = os.path.join(self.original_dir, original_filename)
                if not os.path.exists(original_dest):
                    import shutil
                    shutil.copy2(audio_file, original_dest)
                    logger.info(f"Copied original file to: {original_dest}")
            
            # Process each segment
            segment_data_list = []
            speaker_stats = {}
            source_name = Path(audio_file).stem
            
            for i, segment in enumerate(filtered_segments):
                try:
                    # Create segments directory (flat structure)
                    segments_dir = self.create_segments_directory()
                    
                    # Generate filename
                    segment_filename = self.generate_segment_filename(
                        segment.speaker, i, audio_file
                    )
                    segment_path = os.path.join(segments_dir, segment_filename)
                    
                    # Extract audio segment
                    audio_segment = self.audio_processor.segment_audio(
                        audio, segment.start, segment.end
                    )
                    
                    # Calculate quality score
                    quality_score = self.calculate_quality_score(segment, audio_segment)
                    
                    # Save audio segment
                    self.audio_processor.save_audio(audio_segment, segment_path)
                    
                    # Create segment data with absolute path
                    segment_data = AudioSegmentData(
                        segment_id=f"{source_name}_{segment.speaker}_{i:03d}",
                        start_time=segment.start,
                        end_time=segment.end,
                        speaker=segment.speaker,
                        text=segment.text.strip(),
                        audio_file=os.path.abspath(segment_path),  # Store absolute path
                        source=os.path.basename(audio_file),
                        duration=segment.end - segment.start,
                        confidence=segment.confidence,
                        quality_score=quality_score
                    )
                    
                    segment_data_list.append(segment_data)
                    
                    # Update speaker statistics
                    if segment.speaker not in speaker_stats:
                        speaker_stats[segment.speaker] = {
                            "total_segments": 0,
                            "total_duration": 0.0,
                            "avg_segment_length": 0.0,
                            "avg_confidence": 0.0,
                            "avg_quality_score": 0.0
                        }
                    
                    stats = speaker_stats[segment.speaker]
                    stats["total_segments"] += 1
                    stats["total_duration"] += segment_data.duration
                    
                    logger.debug(f"Created segment {i+1}/{len(filtered_segments)}: "
                               f"{segment.speaker} ({segment_data.duration:.2f}s)")
                    
                except Exception as e:
                    logger.error(f"Failed to process segment {i}: {e}")
                    continue
            
            # Calculate final statistics
            total_duration = sum(seg.duration for seg in segment_data_list)
            
            for speaker, stats in speaker_stats.items():
                if stats["total_segments"] > 0:
                    stats["avg_segment_length"] = stats["total_duration"] / stats["total_segments"]
                    
                    # Calculate average confidence and quality
                    speaker_segments = [s for s in segment_data_list if s.speaker == speaker]
                    if speaker_segments:
                        stats["avg_confidence"] = sum(s.confidence for s in speaker_segments) / len(speaker_segments)
                        stats["avg_quality_score"] = sum(s.quality_score for s in speaker_segments) / len(speaker_segments)
            
            # Create processing info
            processing_info = {
                "total_files": 1,
                "total_segments": len(segment_data_list),
                "processing_time": str(__import__('datetime').datetime.now().isoformat()),
                "service": "assemblyai",
                "api_version": "v2",
                "average_confidence": sum(s.confidence for s in segment_data_list) / len(segment_data_list) if segment_data_list else 0.0,
                "average_quality_score": sum(s.quality_score for s in segment_data_list) / len(segment_data_list) if segment_data_list else 0.0,
                "source_file": audio_file,
                "filtered_segments": len(filtered_segments),
                "original_segments": len(transcription_result.segments)
            }
            
            result = SegmentationResult(
                segments=segment_data_list,
                total_segments=len(segment_data_list),
                total_duration=total_duration,
                speakers=speaker_stats,
                processing_info=processing_info
            )
            
            logger.info(f"Successfully segmented {audio_file}: {len(segment_data_list)} segments, "
                       f"{total_duration:.2f}s total duration")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to segment audio file {audio_file}: {e}")
            raise
    
    def save_metadata(self, result: SegmentationResult, filename: str = None) -> str:
        """
        Save segmentation metadata to JSON file
        
        Args:
            result: SegmentationResult object
            filename: Output filename (optional)
            
        Returns:
            Path to saved metadata file
        """
        try:
            if filename is None:
                filename = f"segments_metadata_{uuid.uuid4().hex[:8]}.json"
            
            metadata_path = os.path.join(self.metadata_dir, filename)
            
            # Convert to dictionary for JSON serialization
            metadata_dict = result.dict()
            
            # Save to JSON
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved metadata to: {metadata_path}")
            return metadata_path
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def load_metadata(self, metadata_path: str) -> SegmentationResult:
        """
        Load segmentation metadata from JSON file
        
        Args:
            metadata_path: Path to metadata file
            
        Returns:
            SegmentationResult object
        """
        try:
            import json
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            result = SegmentationResult(**metadata_dict)
            logger.info(f"Loaded metadata from: {metadata_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load metadata from {metadata_path}: {e}")
            raise
