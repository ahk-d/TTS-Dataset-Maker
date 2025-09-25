"""
Audio segmentation tasks
"""
import os
import librosa
import soundfile as sf
import logging
from typing import List, Dict
from datetime import datetime
from pathlib import Path

from ..models.audio import AudioSegment
from ..models.transcription import DiarizationResult

logger = logging.getLogger(__name__)


async def create_audio_segments(
    audio_file: str,
    diarization_result: DiarizationResult,
    output_dir: str
) -> List[AudioSegment]:
    """Create audio segments from diarization results"""
    # logger = get_run_logger()
    
    try:
        # Load original audio
        audio, sample_rate = librosa.load(audio_file, sr=24000)
        
        segments = []
        utterances = diarization_result.utterances
        
        # Create output directory
        segments_dir = Path(output_dir) / "audio_segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        for i, utterance in enumerate(utterances):
            start_sample = int(utterance.start * sample_rate)
            end_sample = int(utterance.end * sample_rate)
            
            # Extract segment
            segment_audio = audio[start_sample:end_sample]
            
            # Save segment
            segment_filename = f"segment_{i:06d}.wav"
            segment_path = segments_dir / segment_filename
            
            sf.write(segment_path, segment_audio, sample_rate)
            
            # Create segment object
            segment = AudioSegment(
                segment_id=f"seg_{i:06d}",
                source_file_id=os.path.basename(audio_file),
                audio_file_path=str(segment_path),
                text=utterance.text,
                speaker_id=utterance.speaker,
                start_time=utterance.start,
                end_time=utterance.end,
                duration=utterance.end - utterance.start,
                sample_rate=sample_rate,
                quality_score=utterance.confidence,
                confidence_score=utterance.confidence,
                language="en",  # TODO: Get from transcription
                created_at=datetime.now(),
                metadata={
                    "source_file": audio_file,
                    "utterance_index": i,
                    "speaker_confidence": utterance.confidence
                }
            )
            
            segments.append(segment)
        
        logger.info(f"Created {len(segments)} audio segments")
        return segments
        
    except Exception as e:
        logger.error(f"Error creating audio segments: {e}")
        raise


async def validate_segments(segments: List[AudioSegment]) -> List[AudioSegment]:
    """Validate and filter segments based on quality criteria"""
    # logger = get_run_logger()
    
    try:
        from ..config.settings import settings
        
        valid_segments = []
        
        for segment in segments:
            # Check duration
            if segment.duration < settings.min_segment_duration:
                logger.warning(f"Segment {segment.segment_id} too short: {segment.duration:.2f}s")
                continue
            
            if segment.duration > settings.max_segment_duration:
                logger.warning(f"Segment {segment.segment_id} too long: {segment.duration:.2f}s")
                continue
            
            # Check quality score
            if segment.quality_score < settings.min_confidence_score:
                logger.warning(f"Segment {segment.segment_id} low quality: {segment.quality_score:.2f}")
                continue
            
            # Check if audio file exists
            if not os.path.exists(segment.audio_file_path):
                logger.warning(f"Segment {segment.segment_id} audio file missing")
                continue
            
            valid_segments.append(segment)
        
        logger.info(f"Validated {len(valid_segments)}/{len(segments)} segments")
        return valid_segments
        
    except Exception as e:
        logger.error(f"Error validating segments: {e}")
        raise


@task
async def apply_speaker_alignment(
    segments: List[AudioSegment],
    alignment_mapping: Dict[str, str]
) -> List[AudioSegment]:
    """Apply speaker alignment mapping to segments"""
    # logger = get_run_logger()
    
    try:
        aligned_segments = []
        
        for segment in segments:
            # Apply alignment mapping
            original_speaker = segment.speaker_id
            aligned_speaker = alignment_mapping.get(original_speaker, original_speaker)
            
            # Create aligned segment
            aligned_segment = AudioSegment(
                segment_id=segment.segment_id,
                source_file_id=segment.source_file_id,
                audio_file_path=segment.audio_file_path,
                text=segment.text,
                speaker_id=aligned_speaker,
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration=segment.duration,
                sample_rate=segment.sample_rate,
                quality_score=segment.quality_score,
                confidence_score=segment.confidence_score,
                language=segment.language,
                created_at=segment.created_at,
                metadata={
                    **segment.metadata,
                    "original_speaker": original_speaker,
                    "aligned_speaker": aligned_speaker,
                    "alignment_applied": True
                }
            )
            
            aligned_segments.append(aligned_segment)
        
        logger.info(f"Applied speaker alignment to {len(aligned_segments)} segments")
        return aligned_segments
        
    except Exception as e:
        logger.error(f"Error applying speaker alignment: {e}")
        raise
