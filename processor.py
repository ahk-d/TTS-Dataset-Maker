#!/usr/bin/env python3
"""
Simple Local TTS Dataset Processor
No Docker, no Prefect, no async - just pure local processing
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import librosa
import soundfile as sf
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tts_pipeline.integrations.assemblyai import AssemblyAIClient
from tts_pipeline.integrations.label_studio import LabelStudioManager
from tts_pipeline.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Processor:
    """Simple local TTS dataset processor"""
    
    def __init__(self):
        if not settings.assemblyai_api_key:
            raise ValueError("AssemblyAI API key not configured. Please set ASSEMBLYAI_API_KEY environment variable.")
        
        self.assemblyai_client = AssemblyAIClient(settings.assemblyai_api_key)
        self.label_studio = LabelStudioManager()
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "audio_segments").mkdir(exist_ok=True)
        (self.output_dir / "exports").mkdir(exist_ok=True)
    
    def remove_long_silences(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Remove long silences while preserving short gaps"""
        logger.info("Removing long silences...")
        
        # Calculate RMS energy
        hop_length = 512
        frame_length = 2048
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Create speech mask
        speech_mask = rms_db > settings.silence_threshold
        
        # Find speech segments
        speech_segments = []
        in_speech = False
        speech_start = 0
        
        frame_duration = hop_length / sample_rate
        
        for i, is_speech in enumerate(speech_mask):
            if is_speech and not in_speech:
                speech_start = i
                in_speech = True
            elif not is_speech and in_speech:
                speech_segments.append((speech_start, i))
                in_speech = False
        
        # Handle case where audio ends during speech
        if in_speech:
            speech_segments.append((speech_start, len(speech_mask)))
        
        if not speech_segments:
            logger.warning("No speech segments found")
            return audio
        
        # Merge segments that are close together
        merged_segments = []
        current_start, current_end = speech_segments[0]
        
        for start, end in speech_segments[1:]:
            gap_duration = (start - current_end) * frame_duration
            
            if gap_duration <= settings.max_silence_duration:
                # Merge segments (keep the gap)
                current_end = end
            else:
                # Add padding and save current segment
                padding_frames = int(settings.silence_padding_duration / frame_duration)
                merged_segments.append((
                    max(0, current_start - padding_frames),
                    min(len(speech_mask), current_end + padding_frames)
                ))
                current_start, current_end = start, end
        
        # Add the last segment
        padding_frames = int(settings.silence_padding_duration / frame_duration)
        merged_segments.append((
            max(0, current_start - padding_frames),
            min(len(speech_mask), current_end + padding_frames)
        ))
        
        # Extract audio segments
        processed_audio = []
        for start_frame, end_frame in merged_segments:
            start_sample = start_frame * hop_length
            end_sample = end_frame * hop_length
            segment = audio[start_sample:end_sample]
            processed_audio.append(segment)
        
        # Concatenate segments
        if processed_audio:
            result = np.concatenate(processed_audio)
            logger.info(f"Silence removal: {len(audio)/sample_rate:.2f}s -> {len(result)/sample_rate:.2f}s")
            return result
        else:
            return audio
    
    def process_audio_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single audio file"""
        logger.info(f"Processing: {file_path}")
        
        # Load audio
        audio, sample_rate = librosa.load(file_path, sr=24000)
        logger.info(f"Loaded audio: {len(audio)/sample_rate:.2f}s at {sample_rate}Hz")
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Remove long silences if enabled
        if settings.remove_long_silences:
            audio = self.remove_long_silences(audio, sample_rate)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        return {
            "audio": audio,
            "sample_rate": sample_rate,
            "duration": len(audio) / sample_rate
        }
    
    def transcribe_audio(self, audio_file: str) -> Dict:
        """Transcribe audio using AssemblyAI"""
        logger.info("Starting transcription...")
        
        # Use the synchronous method
        transcript = self.assemblyai_client.transcribe_audio(audio_file)
        
        if not transcript:
            raise ValueError("Transcription failed")
        
        logger.info(f"Transcription completed: {len(transcript.utterances)} utterances")
        return transcript
    
    def create_segments(self, transcript: Dict, dataset_name: str, audio_data: Dict) -> List[Dict]:
        """Create audio segments from transcript"""
        logger.info("Creating audio segments...")
        
        segments = []
        utterances = transcript.utterances
        audio = audio_data["audio"]
        sample_rate = audio_data["sample_rate"]
        
        # Create output directory
        segments_dir = self.output_dir / "audio_segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        for i, utterance in enumerate(utterances):
            # Extract audio segment
            start_sample = int(utterance.start * sample_rate / 1000.0)  # Convert ms to samples
            end_sample = int(utterance.end * sample_rate / 1000.0)
            segment_audio = audio[start_sample:end_sample]
            
            # Save audio file
            audio_filename = f"segment_{i:06d}.wav"
            audio_path = segments_dir / audio_filename
            sf.write(audio_path, segment_audio, sample_rate)
            
            # Create segment metadata
            segment = {
                "segment_id": f"seg_{i:06d}",
                "text": utterance.text,
                "speaker_id": utterance.speaker,
                "start_time": utterance.start / 1000.0,  # Convert ms to seconds
                "end_time": utterance.end / 1000.0,
                "duration": (utterance.end - utterance.start) / 1000.0,
                "confidence": utterance.confidence,
                "audio_file": f"audio/{audio_filename}"
            }
            segments.append(segment)
        
        logger.info(f"Created {len(segments)} segments")
        return segments
    
    def validate_segments(self, segments: List[Dict]) -> List[Dict]:
        """Validate and filter segments"""
        logger.info("Validating segments...")
        
        valid_segments = []
        
        for segment in segments:
            # Check duration
            if segment["duration"] < settings.min_segment_duration:
                logger.warning(f"Segment {segment['segment_id']} too short: {segment['duration']:.2f}s")
                continue
            
            if segment["duration"] > settings.max_segment_duration:
                logger.warning(f"Segment {segment['segment_id']} too long: {segment['duration']:.2f}s")
                continue
            
            # Check quality score
            if segment["confidence"] < settings.min_confidence_score:
                logger.warning(f"Segment {segment['segment_id']} low quality: {segment['confidence']:.2f}")
                continue
            
            valid_segments.append(segment)
        
        logger.info(f"Validated {len(valid_segments)}/{len(segments)} segments")
        return valid_segments
    
    def export_dataset(self, segments: List[Dict], dataset_name: str) -> str:
        """Export dataset in Hugging Face format"""
        logger.info(f"Exporting dataset: {dataset_name}")
        
        export_dir = self.output_dir / "exports" / dataset_name
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata.json
        metadata = []
        for segment in segments:
            metadata.append({
                "segment_id": segment["segment_id"],
                "audio_file": segment["audio_file"],
                "text": segment["text"],
                "speaker_id": segment["speaker_id"],
                "duration": segment["duration"],
                "sample_rate": 24000,
                "language": "en",
                "quality_score": segment["confidence"],
                "confidence_score": segment["confidence"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "metadata": {
                    "speaker_confidence": segment["confidence"]
                }
            })
        
        # Save metadata
        with open(export_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create audio directory
        audio_dir = export_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # Create README
        readme_content = f"""# {dataset_name}

TTS Dataset created with local processing pipeline.

## Statistics
- Total segments: {len(segments)}
- Total duration: {sum(s['duration'] for s in segments):.2f}s
- Unique speakers: {len(set(s['speaker_id'] for s in segments))}

## Files
- `metadata.json`: Complete segment metadata
- `audio/`: Individual audio segment files
"""
        
        with open(export_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        logger.info(f"Dataset exported to: {export_dir}")
        return str(export_dir)
    
    def process_dataset(self, file_paths: List[str], dataset_name: str) -> Dict[str, Any]:
        """Process complete dataset"""
        logger.info(f"Processing dataset '{dataset_name}' with {len(file_paths)} files")
        
        start_time = time.time()
        all_segments = []
        
        for i, file_path in enumerate(file_paths):
            logger.info(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
            
            try:
                # Process audio
                audio_data = self.process_audio_file(file_path)
                
                # Transcribe
                transcript = self.transcribe_audio(file_path)
                
                # Create segments
                segments = self.create_segments(transcript, dataset_name, audio_data)
                
                # Validate segments
                valid_segments = self.validate_segments(segments)
                
                all_segments.extend(valid_segments)
                
                logger.info(f"✅ File {i+1} completed: {len(valid_segments)} segments")
                
            except Exception as e:
                logger.error(f"❌ File {i+1} failed: {e}")
        
        if not all_segments:
            raise ValueError("No valid segments created")
        
        # Export dataset
        export_path = self.export_dataset(all_segments, dataset_name)
        
        # Calculate statistics
        total_duration = sum(s["duration"] for s in all_segments)
        unique_speakers = len(set(s["speaker_id"] for s in all_segments))
        
        processing_time = time.time() - start_time
        
        result = {
            "dataset_name": dataset_name,
            "total_files": len(file_paths),
            "total_segments": len(all_segments),
            "total_duration": total_duration,
            "unique_speakers": unique_speakers,
            "processing_time": processing_time,
            "export_path": export_path
        }
        
        logger.info(f"✅ Dataset processing complete!")
        logger.info(f"📊 Segments: {len(all_segments)}")
        logger.info(f"⏱️ Duration: {total_duration:.2f}s")
        logger.info(f"👥 Speakers: {unique_speakers}")
        logger.info(f"📁 Export: {export_path}")
        
        return result

    def process_config_file(self, config_file: str) -> Dict[str, Any]:
        """Process dataset from config file"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        dataset_name = config['name']
        sources = config.get('sources', [])
        
        return self.process_dataset(sources, dataset_name)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Local TTS Dataset Processor")
    parser.add_argument("config", help="JSON configuration file")
    parser.add_argument("--output", help="Output directory", default="output")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        dataset_name = config['name']
        sources = config.get('sources', [])
        
        # Separate local files from URLs
        local_files = []
        urls = []
        
        for source in sources:
            if source.startswith(('http://', 'https://')):
                urls.append(source)
            else:
                local_files.append(source)
        
        if urls:
            logger.warning(f"URL processing not implemented in local mode. Skipping {len(urls)} URLs.")
        
        if not local_files:
            raise ValueError("No local files to process")
        
        # Process dataset
        processor = LocalTTSProcessor()
        result = processor.process_dataset(local_files, dataset_name)
        
        print(f"\n🎉 Processing Complete!")
        print(f"📊 Dataset: {result['dataset_name']}")
        print(f"📁 Files: {result['total_files']}")
        print(f"🎵 Segments: {result['total_segments']}")
        print(f"⏱️ Duration: {result['total_duration']:.2f}s")
        print(f"👥 Speakers: {result['unique_speakers']}")
        print(f"📁 Export: {result['export_path']}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
