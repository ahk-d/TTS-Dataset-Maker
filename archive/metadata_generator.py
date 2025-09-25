"""
Metadata generation module for TTS Dataset Maker
Creates structured metadata for TTS training data compatible with Orpheus TTS
"""
import os
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from audio_segmenter import SegmentationResult, AudioSegmentData
from config import settings


logger = logging.getLogger(__name__)


class TTSMetadata(BaseModel):
    """Metadata for TTS training data"""
    segment_id: str
    audio_file: str
    text: str
    speaker_id: str
    duration: float
    sample_rate: int = Field(default=24000)
    language: str = Field(default="en")
    quality_score: float
    confidence_score: float
    source_file: str
    start_time: float
    end_time: float
    metadata_version: str = Field(default="1.0")


class DatasetMetadata(BaseModel):
    """Complete dataset metadata"""
    dataset_name: str
    version: str
    created_at: str
    total_segments: int
    total_duration: float
    speakers: Dict[str, Dict[str, Any]]
    processing_info: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    format_compatibility: Dict[str, str]


class HuggingFaceDatasetConfig(BaseModel):
    """HuggingFace dataset configuration"""
    dataset_name: str
    version: str
    features: Dict[str, Any]
    splits: Dict[str, int]
    download_size: str
    dataset_size: str


class MetadataGenerator:
    """Generates metadata for TTS training datasets"""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize metadata generator
        
        Args:
            output_dir: Output directory for metadata files
        """
        self.output_dir = output_dir or settings.output_dir
        self.metadata_dir = os.path.join(self.output_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        logger.info(f"Metadata generator initialized with output dir: {self.output_dir}")
    
    def convert_to_tts_metadata(self, segment_data: AudioSegmentData, 
                               language: str = "en") -> TTSMetadata:
        """
        Convert AudioSegmentData to TTSMetadata format
        
        Args:
            segment_data: Audio segment data
            language: Language code
            
        Returns:
            TTSMetadata object
        """
        return TTSMetadata(
            segment_id=segment_data.segment_id,
            audio_file=segment_data.audio_file,
            text=segment_data.text,
            speaker_id=segment_data.speaker,
            duration=segment_data.duration,
            sample_rate=settings.default_sample_rate,
            language=language,
            quality_score=segment_data.quality_score or 0.0,
            confidence_score=segment_data.confidence,
            source_file=segment_data.source,
            start_time=segment_data.start_time,
            end_time=segment_data.end_time
        )
    
    def generate_dataset_metadata(self, segmentation_results: List[SegmentationResult],
                                 dataset_name: str = None) -> DatasetMetadata:
        """
        Generate complete dataset metadata
        
        Args:
            segmentation_results: List of segmentation results
            dataset_name: Name for the dataset
            
        Returns:
            DatasetMetadata object
        """
        if dataset_name is None:
            dataset_name = f"tts_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Aggregate data from all results
        all_segments = []
        all_speakers = {}
        total_duration = 0.0
        total_files = 0
        
        for result in segmentation_results:
            all_segments.extend(result.segments)
            total_duration += result.total_duration
            total_files += result.processing_info.get("total_files", 1)
            
            # Merge speaker statistics
            for speaker, stats in result.speakers.items():
                if speaker not in all_speakers:
                    all_speakers[speaker] = {
                        "total_segments": 0,
                        "total_duration": 0.0,
                        "avg_segment_length": 0.0,
                        "avg_confidence": 0.0,
                        "avg_quality_score": 0.0,
                        "files": set()
                    }
                
                all_speakers[speaker]["total_segments"] += stats["total_segments"]
                all_speakers[speaker]["total_duration"] += stats["total_duration"]
                all_speakers[speaker]["files"].add(result.processing_info.get("source_file", "unknown"))
        
        # Calculate final speaker statistics
        for speaker, stats in all_speakers.items():
            if stats["total_segments"] > 0:
                stats["avg_segment_length"] = stats["total_duration"] / stats["total_segments"]
                
                # Calculate averages from segments
                speaker_segments = [s for s in all_segments if s.speaker == speaker]
                if speaker_segments:
                    stats["avg_confidence"] = sum(s.confidence for s in speaker_segments) / len(speaker_segments)
                    stats["avg_quality_score"] = sum(s.quality_score or 0.0 for s in speaker_segments) / len(speaker_segments)
                
                # Convert files set to list for JSON serialization
                stats["files"] = list(stats["files"])
        
        # Calculate quality metrics
        quality_metrics = self.calculate_quality_metrics(all_segments)
        
        # Create processing info
        processing_info = {
            "total_files": total_files,
            "total_segments": len(all_segments),
            "processing_time": datetime.now().isoformat(),
            "service": "assemblyai",
            "api_version": "v2",
            "average_confidence": sum(s.confidence for s in all_segments) / len(all_segments) if all_segments else 0.0,
            "average_quality_score": sum(s.quality_score or 0.0 for s in all_segments) / len(all_segments) if all_segments else 0.0,
            "total_duration_hours": total_duration / 3600.0,
            "speakers_count": len(all_speakers)
        }
        
        # Format compatibility info
        format_compatibility = {
            "orpheus_tts": "compatible",
            "huggingface_datasets": "compatible",
            "torchaudio": "compatible",
            "sample_rate": f"{settings.default_sample_rate}Hz",
            "format": "WAV",
            "channels": "mono"
        }
        
        return DatasetMetadata(
            dataset_name=dataset_name,
            version="1.0",
            created_at=datetime.now().isoformat(),
            total_segments=len(all_segments),
            total_duration=total_duration,
            speakers=all_speakers,
            processing_info=processing_info,
            quality_metrics=quality_metrics,
            format_compatibility=format_compatibility
        )
    
    def calculate_quality_metrics(self, segments: List[AudioSegmentData]) -> Dict[str, Any]:
        """
        Calculate quality metrics for the dataset
        
        Args:
            segments: List of audio segments
            
        Returns:
            Dictionary with quality metrics
        """
        if not segments:
            return {}
        
        # Duration statistics
        durations = [s.duration for s in segments]
        confidences = [s.confidence for s in segments]
        quality_scores = [s.quality_score or 0.0 for s in segments]
        text_lengths = [len(s.text) for s in segments]
        
        metrics = {
            "duration_stats": {
                "min": min(durations),
                "max": max(durations),
                "mean": sum(durations) / len(durations),
                "median": sorted(durations)[len(durations) // 2]
            },
            "confidence_stats": {
                "min": min(confidences),
                "max": max(confidences),
                "mean": sum(confidences) / len(confidences),
                "median": sorted(confidences)[len(confidences) // 2]
            },
            "quality_score_stats": {
                "min": min(quality_scores),
                "max": max(quality_scores),
                "mean": sum(quality_scores) / len(quality_scores),
                "median": sorted(quality_scores)[len(quality_scores) // 2]
            },
            "text_length_stats": {
                "min": min(text_lengths),
                "max": max(text_lengths),
                "mean": sum(text_lengths) / len(text_lengths),
                "median": sorted(text_lengths)[len(text_lengths) // 2]
            },
            "quality_distribution": {
                "high_quality": len([s for s in segments if (s.quality_score or 0.0) > 0.8]),
                "medium_quality": len([s for s in segments if 0.6 <= (s.quality_score or 0.0) <= 0.8]),
                "low_quality": len([s for s in segments if (s.quality_score or 0.0) < 0.6])
            },
            "confidence_distribution": {
                "high_confidence": len([s for s in segments if s.confidence > 0.8]),
                "medium_confidence": len([s for s in segments if 0.6 <= s.confidence <= 0.8]),
                "low_confidence": len([s for s in segments if s.confidence < 0.6])
            }
        }
        
        return metrics
    
    def create_huggingface_dataset_config(self, dataset_metadata: DatasetMetadata) -> HuggingFaceDatasetConfig:
        """
        Create HuggingFace dataset configuration
        
        Args:
            dataset_metadata: Dataset metadata
            
        Returns:
            HuggingFaceDatasetConfig object
        """
        features = {
            "audio": {
                "sampling_rate": settings.default_sample_rate,
                "dtype": "float32",
                "id": None,
                "_type": "Audio"
            },
            "text": {
                "dtype": "string",
                "id": None,
                "_type": "Value"
            },
            "speaker_id": {
                "dtype": "string",
                "id": None,
                "_type": "Value"
            },
            "duration": {
                "dtype": "float64",
                "id": None,
                "_type": "Value"
            },
            "quality_score": {
                "dtype": "float64",
                "id": None,
                "_type": "Value"
            }
        }
        
        # Calculate dataset size (rough estimate)
        avg_duration = dataset_metadata.total_duration / dataset_metadata.total_segments if dataset_metadata.total_segments > 0 else 0
        estimated_size_mb = (dataset_metadata.total_segments * avg_duration * settings.default_sample_rate * 2) / (1024 * 1024)
        
        return HuggingFaceDatasetConfig(
            dataset_name=dataset_metadata.dataset_name,
            version=dataset_metadata.version,
            features=features,
            splits={"train": dataset_metadata.total_segments},
            download_size=f"{estimated_size_mb:.1f}MB",
            dataset_size=f"{estimated_size_mb:.1f}MB"
        )
    
    def save_tts_metadata(self, segments: List[TTSMetadata], filename: str = None) -> str:
        """
        Save TTS metadata to JSON file
        
        Args:
            segments: List of TTS metadata
            filename: Output filename (optional)
            
        Returns:
            Path to saved metadata file
        """
        try:
            if filename is None:
                filename = f"tts_metadata_{uuid.uuid4().hex[:8]}.json"
            
            metadata_path = os.path.join(self.metadata_dir, filename)
            
            # Convert to list of dictionaries
            metadata_list = [segment.dict() for segment in segments]
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved TTS metadata to: {metadata_path}")
            return metadata_path
            
        except Exception as e:
            logger.error(f"Failed to save TTS metadata: {e}")
            raise
    
    def save_dataset_metadata(self, dataset_metadata: DatasetMetadata, 
                             filename: str = None) -> str:
        """
        Save dataset metadata to JSON file
        
        Args:
            dataset_metadata: Dataset metadata
            filename: Output filename (optional)
            
        Returns:
            Path to saved metadata file
        """
        try:
            if filename is None:
                filename = f"dataset_metadata_{dataset_metadata.dataset_name}.json"
            
            metadata_path = os.path.join(self.metadata_dir, filename)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_metadata.dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved dataset metadata to: {metadata_path}")
            return metadata_path
            
        except Exception as e:
            logger.error(f"Failed to save dataset metadata: {e}")
            raise
    
    def save_huggingface_config(self, hf_config: HuggingFaceDatasetConfig,
                               filename: str = None) -> str:
        """
        Save HuggingFace dataset configuration
        
        Args:
            hf_config: HuggingFace dataset configuration
            filename: Output filename (optional)
            
        Returns:
            Path to saved config file
        """
        try:
            if filename is None:
                filename = f"huggingface_config_{hf_config.dataset_name}.json"
            
            config_path = os.path.join(self.metadata_dir, filename)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(hf_config.dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved HuggingFace config to: {config_path}")
            return config_path
            
        except Exception as e:
            logger.error(f"Failed to save HuggingFace config: {e}")
            raise
    
    def create_orpheus_training_config(self, dataset_metadata: DatasetMetadata,
                                      output_path: str = None) -> str:
        """
        Create Orpheus TTS training configuration
        
        Args:
            dataset_metadata: Dataset metadata
            output_path: Output path for config file
            
        Returns:
            Path to saved config file
        """
        try:
            if output_path is None:
                output_path = os.path.join(self.metadata_dir, f"orpheus_config_{dataset_metadata.dataset_name}.yaml")
            
            config_content = f"""# Orpheus TTS Training Configuration
# Generated for dataset: {dataset_metadata.dataset_name}

dataset:
  name: "{dataset_metadata.dataset_name}"
  version: "{dataset_metadata.version}"
  total_segments: {dataset_metadata.total_segments}
  total_duration: {dataset_metadata.total_duration:.2f}
  speakers: {len(dataset_metadata.speakers)}
  
audio:
  sample_rate: {settings.default_sample_rate}
  format: "wav"
  channels: 1
  
training:
  batch_size: 32
  learning_rate: 1e-4
  epochs: 100
  validation_split: 0.1
  
quality:
  min_confidence: {settings.min_confidence_score}
  min_quality_score: 0.6
  min_duration: {settings.min_segment_duration}
  max_duration: {settings.max_segment_duration}
  
speakers:
"""
            
            for speaker_id, stats in dataset_metadata.speakers.items():
                config_content += f"""  {speaker_id}:
    segments: {stats['total_segments']}
    duration: {stats['total_duration']:.2f}
    avg_confidence: {stats['avg_confidence']:.3f}
    avg_quality: {stats['avg_quality_score']:.3f}
"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            logger.info(f"Saved Orpheus training config to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create Orpheus training config: {e}")
            raise
