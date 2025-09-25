"""
Main TTS Service orchestrator
Coordinates the complete pipeline for TTS dataset creation
"""
import os
import logging
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import argparse
from datetime import datetime

from pydantic import BaseModel

from config import settings
from audio_processor import AudioProcessor
from assemblyai_client import AssemblyAIClient, TranscriptionResult
from audio_segmenter import AudioSegmenter, SegmentationResult
from metadata_generator import MetadataGenerator, TTSMetadata, DatasetMetadata
from quality_control import QualityFilter, QualityReport
from production_storage import ProductionStorage, SourceFile, AudioSegment


logger = logging.getLogger(__name__)


class ProcessingStats(BaseModel):
    """Processing statistics"""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_segments: int = 0
    total_duration: float = 0.0
    processing_time: float = 0.0
    start_time: str = ""
    end_time: str = ""


class TTSDatasetService:
    """Main TTS dataset creation service"""
    
    def __init__(self, output_dir: str = None, api_key: str = None):
        """
        Initialize TTS dataset service with production storage
        
        Args:
            output_dir: Output directory for processed data
            api_key: AssemblyAI API key
        """
        self.output_dir = output_dir or settings.output_dir
        self.api_key = api_key or settings.assemblyai_api_key
        
        # Initialize production storage
        self.storage = ProductionStorage(self.output_dir)
        
        # Initialize components
        self.audio_processor = AudioProcessor()
        self.assemblyai_client = AssemblyAIClient(self.api_key)
        self.audio_segmenter = AudioSegmenter(self.output_dir)
        self.metadata_generator = MetadataGenerator(self.output_dir)
        self.quality_filter = QualityFilter()
        
        # Processing statistics
        self.stats = ProcessingStats()
        
        logger.info(f"TTS Dataset Service initialized with production storage: {self.output_dir}")
    
    def process_single_file(self, audio_file: str, language: str = "en") -> Tuple[bool, SegmentationResult, str]:
        """
        Process a single audio file through the complete pipeline
        This is a convenience method that wraps process_multiple_files for single files
        
        Args:
            audio_file: Path to audio file
            language: Language code for transcription
            
        Returns:
            Tuple of (success, segmentation_result, error_message)
        """
        # Use multi-file processing for single file (multi-file first approach)
        segmentation_results, stats = self.process_multiple_files([audio_file], language)
        
        if stats.processed_files > 0:
            return True, segmentation_results[0], ""
        else:
            return False, None, f"Failed to process {audio_file}"
    
    def process_multiple_files(self, audio_files: List[str], language: str = "en") -> Tuple[List[SegmentationResult], ProcessingStats]:
        """
        Process multiple audio files - this is the core processing method
        
        Args:
            audio_files: List of audio file paths
            language: Language code for transcription
            
        Returns:
            Tuple of (segmentation_results, processing_stats)
        """
        logger.info(f"Processing {len(audio_files)} audio files")
        
        # Initialize statistics
        self.stats = ProcessingStats(
            total_files=len(audio_files),
            start_time=datetime.now().isoformat()
        )
        
        start_time = time.time()
        segmentation_results = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                logger.info(f"Processing file {i+1}/{len(audio_files)}: {audio_file}")
                
                # Process individual file through the pipeline
                success, result, error = self._process_individual_file(audio_file, language)
                
                if success:
                    segmentation_results.append(result)
                    self.stats.processed_files += 1
                    self.stats.total_segments += result.total_segments
                    self.stats.total_duration += result.total_duration
                    logger.info(f"Successfully processed: {audio_file}")
                else:
                    self.stats.failed_files += 1
                    logger.error(f"Failed to process {audio_file}: {error}")
                
            except Exception as e:
                self.stats.failed_files += 1
                logger.error(f"Unexpected error processing {audio_file}: {e}")
        
        # Finalize statistics
        self.stats.processing_time = time.time() - start_time
        self.stats.end_time = datetime.now().isoformat()
        
        logger.info(f"Processing completed: {self.stats.processed_files}/{self.stats.total_files} files successful")
        
        return segmentation_results, self.stats
    
    def _process_individual_file(self, audio_file: str, language: str = "en") -> Tuple[bool, SegmentationResult, str]:
        """
        Process a single audio file through the complete pipeline with production storage
        This is the core processing logic used by both single and multi-file processing
        
        Args:
            audio_file: Path to audio file
            language: Language code for transcription
            
        Returns:
            Tuple of (success, segmentation_result, error_message)
        """
        try:
            logger.info(f"Processing audio file: {audio_file}")
            
            # Register source file in production storage
            file_id = self.storage.register_source_file(audio_file, "audio")
            logger.info(f"Registered source file: {file_id}")
            
            # Step 1: Validate and preprocess audio
            logger.info("Step 1: Audio preprocessing")
            audio, is_valid, issues = self.audio_processor.preprocess_audio(audio_file)
            
            if not is_valid:
                logger.warning(f"Audio quality issues: {issues}")
                self.storage.update_source_file_metadata(file_id, 0, 0, 0, "failed")
                return False, None, f"Audio quality issues: {issues}"
            
            # Step 2: Transcribe with speaker diarization using original audio
            logger.info("Step 2: Transcription and speaker diarization")
            transcription_result = self.assemblyai_client.process_audio_file(audio_file, language)
            
            # Validate transcription quality
            transcription_valid, transcription_issues = self.assemblyai_client.validate_transcription_quality(transcription_result)
            
            if not transcription_valid:
                logger.warning(f"Transcription quality issues: {transcription_issues}")
            
            # Step 3: Segment audio based on speaker diarization using original audio
            logger.info("Step 3: Audio segmentation")
            segmentation_result = self.audio_segmenter.segment_audio_file(audio_file, transcription_result)
            
            if segmentation_result.total_segments == 0:
                self.storage.update_source_file_metadata(file_id, 0, 0, 0, "failed")
                return False, segmentation_result, "No valid segments created"
            
            # Step 4: Quality control and filtering
            logger.info("Step 4: Quality control")
            quality_report = self.quality_filter.generate_quality_report(segmentation_result)
            
            # Filter segments based on quality
            filtered_segments, _ = self.quality_filter.filter_segments(segmentation_result.segments)
            
            if not filtered_segments:
                self.storage.update_source_file_metadata(file_id, 0, 0, 0, "failed")
                return False, segmentation_result, "No segments passed quality filtering"
            
            # Update segmentation result with filtered segments
            segmentation_result.segments = filtered_segments
            segmentation_result.total_segments = len(filtered_segments)
            segmentation_result.total_duration = sum(seg.duration for seg in filtered_segments)
            
            # Update source file metadata with processing results
            self.storage.update_source_file_metadata(
                file_id, 
                segmentation_result.total_duration,
                24000,  # sample rate
                1,      # channels (mono)
                "completed"
            )
            
            # Save audio segments to production storage
            segment_data = []
            for segment in filtered_segments:
                segment_data.append({
                    'audio_file': segment.audio_file,
                    'text': segment.text,
                    'speaker_id': segment.speaker,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'duration': segment.duration,
                    'sample_rate': 24000,
                    'quality_score': segment.quality_score or 0.0,
                    'confidence_score': segment.confidence or 0.0,
                    'language': language,
                    'metadata': {
                        'source_file_id': file_id,
                        'original_file': audio_file,
                        'processing_timestamp': datetime.now().isoformat()
                    }
                })
            
            # Save segments to production storage
            segment_ids = self.storage.save_audio_segments(file_id, segment_data)
            logger.info(f"Saved {len(segment_ids)} segments to production storage")
            
            logger.info(f"Successfully processed {audio_file}: {len(filtered_segments)} segments")
            return True, segmentation_result, ""
            
        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")
            # Update source file status to failed
            try:
                self.storage.update_source_file_metadata(file_id, 0, 0, 0, "failed")
            except:
                pass
            return False, None, str(e)
    
    def create_dataset(self, segmentation_results: List[SegmentationResult], 
                      dataset_name: str = None) -> Tuple[DatasetMetadata, str]:
        """
        Create complete dataset from segmentation results
        
        Args:
            segmentation_results: List of segmentation results
            dataset_name: Name for the dataset
            
        Returns:
            Tuple of (dataset_metadata, output_path)
        """
        try:
            logger.info(f"Creating dataset from {len(segmentation_results)} segmentation results")
            
            # Generate dataset metadata
            dataset_metadata = self.metadata_generator.generate_dataset_metadata(
                segmentation_results, dataset_name
            )
            
            # Save dataset metadata
            metadata_path = self.metadata_generator.save_dataset_metadata(dataset_metadata)
            
            # Convert segments to TTS metadata format
            all_segments = []
            for result in segmentation_results:
                for segment in result.segments:
                    tts_metadata = self.metadata_generator.convert_to_tts_metadata(segment)
                    all_segments.append(tts_metadata)
            
            # Save TTS metadata
            tts_metadata_path = self.metadata_generator.save_tts_metadata(all_segments)
            
            # Create HuggingFace dataset configuration
            hf_config = self.metadata_generator.create_huggingface_dataset_config(dataset_metadata)
            hf_config_path = self.metadata_generator.save_huggingface_config(hf_config)
            
            # Create Orpheus training configuration
            orpheus_config_path = self.metadata_generator.create_orpheus_training_config(dataset_metadata)
            
            logger.info(f"Dataset created successfully: {dataset_metadata.dataset_name}")
            logger.info(f"Total segments: {dataset_metadata.total_segments}")
            logger.info(f"Total duration: {dataset_metadata.total_duration:.2f}s")
            logger.info(f"Speakers: {len(dataset_metadata.speakers)}")
            
            return dataset_metadata, self.output_dir
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise
    
    def run_complete_pipeline(self, audio_files: List[str], dataset_name: str = None, 
                            language: str = "en") -> Tuple[DatasetMetadata, ProcessingStats]:
        """
        Run the complete TTS dataset creation pipeline
        
        Args:
            audio_files: List of audio file paths
            dataset_name: Name for the dataset
            language: Language code for transcription
            
        Returns:
            Tuple of (dataset_metadata, processing_stats)
        """
        try:
            logger.info("Starting complete TTS dataset creation pipeline")
            
            # Process all audio files
            segmentation_results, processing_stats = self.process_multiple_files(audio_files, language)
            
            if not segmentation_results:
                raise Exception("No files were successfully processed")
            
            # Create dataset
            dataset_metadata, output_path = self.create_dataset(segmentation_results, dataset_name)
            
            # Generate final quality report
            all_segments = []
            for result in segmentation_results:
                all_segments.extend(result.segments)
            
            quality_report = self.quality_filter.generate_quality_report(
                SegmentationResult(
                    segments=all_segments,
                    total_segments=len(all_segments),
                    total_duration=sum(seg.duration for seg in all_segments),
                    speakers={},
                    processing_info={}
                )
            )
            
            # Save quality report
            quality_report_path = self.quality_filter.save_quality_report(quality_report)
            
            logger.info("Complete pipeline finished successfully")
            logger.info(f"Output directory: {output_path}")
            logger.info(f"Dataset: {dataset_metadata.dataset_name}")
            logger.info(f"Quality report: {quality_report_path}")
            
            return dataset_metadata, processing_stats
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def get_processing_summary(self) -> Dict[str, any]:
        """
        Get processing summary
        
        Returns:
            Dictionary with processing summary
        """
        return {
            "total_files": self.stats.total_files,
            "processed_files": self.stats.processed_files,
            "failed_files": self.stats.failed_files,
            "success_rate": self.stats.processed_files / self.stats.total_files if self.stats.total_files > 0 else 0,
            "total_segments": self.stats.total_segments,
            "total_duration": self.stats.total_duration,
            "processing_time": self.stats.processing_time,
            "start_time": self.stats.start_time,
            "end_time": self.stats.end_time
        }
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics from production storage"""
        return self.storage.get_dataset_statistics()
    
    def get_source_files(self, status: str = None) -> List[SourceFile]:
        """Get source files with optional status filter"""
        return self.storage.get_source_files(status)
    
    def get_audio_segments(self, source_file_id: str = None, speaker_id: str = None) -> List[AudioSegment]:
        """Get audio segments with optional filters"""
        return self.storage.get_audio_segments(source_file_id, speaker_id)
    
    def create_dataset_export(self, dataset_name: str = None, output_format: str = "huggingface") -> str:
        """Create dataset export in specified format"""
        return self.storage.create_dataset_export(dataset_name, output_format)
    
    def list_speakers(self) -> List[str]:
        """Get list of all speakers in the dataset"""
        segments = self.storage.get_audio_segments()
        return list(set(seg.speaker_id for seg in segments))
    
    def get_speaker_statistics(self, speaker_id: str) -> Dict[str, Any]:
        """Get statistics for a specific speaker"""
        segments = self.storage.get_audio_segments(speaker_id=speaker_id)
        
        if not segments:
            return {"error": f"Speaker '{speaker_id}' not found"}
        
        total_duration = sum(seg.duration for seg in segments)
        avg_quality = sum(seg.quality_score for seg in segments) / len(segments)
        avg_confidence = sum(seg.confidence_score for seg in segments) / len(segments)
        
        return {
            "speaker_id": speaker_id,
            "total_segments": len(segments),
            "total_duration_seconds": total_duration,
            "average_quality_score": avg_quality,
            "average_confidence_score": avg_confidence,
            "source_files": list(set(seg.source_file_id for seg in segments))
        }


def main():
    """Main entry point for command-line usage - Multi-file first design"""
    parser = argparse.ArgumentParser(description="TTS Dataset Maker - Multi-file Processing")
    parser.add_argument("audio_files", nargs="+", help="Audio files to process (supports multiple files)")
    parser.add_argument("--output-dir", "-o", default="./output", help="Output directory")
    parser.add_argument("--dataset-name", "-n", help="Dataset name")
    parser.add_argument("--language", "-l", default="en", help="Language code")
    parser.add_argument("--api-key", help="AssemblyAI API key")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    # Multi-file processing options
    parser.add_argument("--batch-size", type=int, default=1, help="Number of files to process in each batch")
    parser.add_argument("--preserve-temp", action="store_true", help="Preserve temporary files for debugging")
    parser.add_argument("--speaker-mapping", choices=["interactive", "auto", "manual"], 
                       default="interactive", help="Speaker mapping mode")
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive speaker mapping")
    
    # VAD arguments
    parser.add_argument("--vad", action="store_true", help="Enable voice activity detection")
    parser.add_argument("--vad-method", default="webrtcvad",
                       choices=["webrtcvad", "librosa"], help="VAD method")
    parser.add_argument("--min-speech-duration", type=float, default=0.5,
                       help="Minimum speech duration in seconds")
    parser.add_argument("--min-speech-ratio", type=float, default=0.3,
                       help="Minimum speech ratio in segment (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Apply command line arguments to settings
    if args.vad:
        settings.vad_enabled = True
        settings.vad_method = args.vad_method
        settings.min_speech_duration = args.min_speech_duration
        settings.min_speech_ratio = args.min_speech_ratio
    
    # Apply multi-file settings
    settings.batch_size = args.batch_size
    settings.preserve_temp_files = args.preserve_temp
    settings.speaker_mapping_mode = args.speaker_mapping if not args.no_interactive else "auto"
    
    try:
        # Initialize service
        service = TTSDatasetService(args.output_dir, args.api_key)
        
        # Run pipeline
        dataset_metadata, processing_stats = service.run_complete_pipeline(
            args.audio_files, args.dataset_name, args.language
        )
        
        # Print summary
        print("\n" + "="*60)
        print("🎉 TTS DATASET CREATION COMPLETED")
        print("="*60)
        print(f"📊 Dataset Name: {dataset_metadata.dataset_name}")
        print(f"📁 Files Processed: {processing_stats.processed_files}/{processing_stats.total_files}")
        print(f"🎵 Total Segments: {dataset_metadata.total_segments}")
        print(f"⏱️  Total Duration: {dataset_metadata.total_duration:.2f}s ({dataset_metadata.total_duration/3600:.2f}h)")
        print(f"👥 Speakers: {len(dataset_metadata.speakers)}")
        print(f"📂 Output Directory: {args.output_dir}")
        print(f"⏰ Processing Time: {processing_stats.processing_time:.2f}s")
        print(f"✅ Success Rate: {(processing_stats.processed_files/processing_stats.total_files)*100:.1f}%")
        
        if processing_stats.failed_files > 0:
            print(f"❌ Failed Files: {processing_stats.failed_files}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
