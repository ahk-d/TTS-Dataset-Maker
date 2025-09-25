"""
Dataset export tasks
"""
import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from ..models.audio import AudioSegment
from ..models.dataset import DatasetMetadata

logger = logging.getLogger(__name__)


async def create_dataset_metadata(
    segments: List[AudioSegment],
    dataset_name: str,
    output_dir: str
) -> DatasetMetadata:
    """Create dataset metadata from segments"""
    # logger = get_run_logger()
    
    try:
        # Calculate statistics
        total_segments = len(segments)
        total_duration = sum(seg.duration for seg in segments)
        speakers = list(set(seg.speaker_id for seg in segments))
        languages = list(set(seg.language for seg in segments))
        
        metadata = DatasetMetadata(
            dataset_name=dataset_name,
            total_segments=total_segments,
            total_duration=total_duration,
            speakers=speakers,
            languages=languages,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version="1.0",
            description=f"TTS dataset with {total_segments} segments from {len(speakers)} speakers",
            metadata={
                "created_by": "tts-dataset-pipeline",
                "pipeline_version": "0.1.0",
                "total_files": len(set(seg.source_file_id for seg in segments)),
                "average_quality": sum(seg.quality_score for seg in segments) / total_segments if total_segments > 0 else 0.0,
                "average_confidence": sum(seg.confidence_score for seg in segments) / total_segments if total_segments > 0 else 0.0
            }
        )
        
        logger.info(f"Created dataset metadata: {dataset_name}")
        return metadata
        
    except Exception as e:
        logger.error(f"Error creating dataset metadata: {e}")
        raise


async def export_to_huggingface(
    segments: List[AudioSegment],
    dataset_metadata: DatasetMetadata,
    output_dir: str
) -> str:
    """Export dataset to Hugging Face format"""
    # logger = get_run_logger()
    
    try:
        # Create export directory
        export_dir = Path(output_dir) / "exports" / dataset_metadata.dataset_name
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Create audio directory
        audio_dir = export_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # Copy audio files and create metadata
        metadata = []
        for i, segment in enumerate(segments):
            # Copy audio file
            new_audio_path = audio_dir / f"segment_{i:06d}.wav"
            if Path(segment.audio_file_path).exists():
                shutil.copy2(segment.audio_file_path, new_audio_path)
            
            # Create metadata entry
            metadata.append({
                "segment_id": segment.segment_id,
                "audio_file": str(new_audio_path),
                "text": segment.text,
                "speaker_id": segment.speaker_id,
                "duration": segment.duration,
                "sample_rate": segment.sample_rate,
                "language": segment.language,
                "quality_score": segment.quality_score,
                "confidence_score": segment.confidence_score,
                "source_file_id": segment.source_file_id,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "metadata": segment.metadata
            })
        
        # Save metadata
        metadata_file = export_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Create dataset card
        await create_dataset_card(export_dir, dataset_metadata)
        
        logger.info(f"Hugging Face export completed: {export_dir}")
        return str(export_dir)
        
    except Exception as e:
        logger.error(f"Error exporting to Hugging Face: {e}")
        raise


@task
async def export_to_orpheus(
    segments: List[AudioSegment],
    dataset_metadata: DatasetMetadata,
    output_dir: str
) -> str:
    """Export dataset to Orpheus TTS format"""
    # logger = get_run_logger()
    
    try:
        # Create export directory
        export_dir = Path(output_dir) / "exports" / dataset_metadata.dataset_name
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Create audio directory
        audio_dir = export_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # Create Orpheus metadata format
        orpheus_metadata = {
            "dataset_name": dataset_metadata.dataset_name,
            "total_segments": len(segments),
            "speakers": list(set(seg.speaker_id for seg in segments)),
            "segments": []
        }
        
        for i, segment in enumerate(segments):
            # Copy audio file
            new_audio_path = audio_dir / f"segment_{i:06d}.wav"
            if Path(segment.audio_file_path).exists():
                shutil.copy2(segment.audio_file_path, new_audio_path)
            
            # Add to Orpheus format
            orpheus_metadata["segments"].append({
                "audio_file": str(new_audio_path),
                "text": segment.text,
                "speaker": segment.speaker_id,
                "duration": segment.duration
            })
        
        # Save Orpheus metadata
        metadata_file = export_dir / "orpheus_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(orpheus_metadata, f, indent=2, default=str)
        
        logger.info(f"Orpheus export completed: {export_dir}")
        return str(export_dir)
        
    except Exception as e:
        logger.error(f"Error exporting to Orpheus: {e}")
        raise


@task
async def create_dataset_card(export_dir: Path, dataset_metadata: DatasetMetadata):
    """Create dataset card for Hugging Face"""
    
    card_content = f"""---
language:
- en
pretty_name: "{dataset_metadata.dataset_name}"
tags:
- audio
- speech
- tts
- transcription
- english
license: "mit"
task_categories:
- text-to-speech
- automatic-speech-recognition
size_categories:
- 1K<n<10K
---

# {dataset_metadata.dataset_name}

This dataset contains {dataset_metadata.total_segments} audio segments with transcriptions for Text-to-Speech training.

## Dataset Structure
- `audio/`: Audio files (WAV format, 24kHz)
- `metadata.json`: Contains segment metadata

## Usage
```python
import json

# Load metadata
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

# Access segments
for segment in metadata:
    print(f"Text: {{segment['text']}}")
    print(f"Speaker: {{segment['speaker_id']}}")
    print(f"Audio: {{segment['audio_file']}}")
```

## Statistics
- Total segments: {dataset_metadata.total_segments}
- Total duration: {dataset_metadata.total_duration:.1f}s
- Speakers: {len(dataset_metadata.speakers)}
- Languages: {', '.join(dataset_metadata.languages)}
- Sample rate: 24kHz
- Format: WAV
"""
    
    readme_file = export_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(card_content)
