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
import shutil
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
from tts_pipeline.tasks.audio_processing import preprocess_audio, denoise_with_deepfilternet

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
        (self.output_dir / "transcripts").mkdir(exist_ok=True)
    
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
        
        # Load original audio at native SR for precise timestamp alignment
        audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
        audio = librosa.util.normalize(audio)

        # Apply denoising if enabled (but not silence removal to preserve timing)
        audio_denoised = None
        denoising_metrics = {}
        try:
            if settings.denoising_enabled and os.environ.get("DISABLE_DEEPFILTERNET", "false").lower() != "true":
                denoised, denoising_metrics = denoise_with_deepfilternet(audio, sample_rate)
                if isinstance(denoised, np.ndarray) and len(denoised) == len(audio):
                    audio_denoised = denoised
                    logger.info("Using denoised audio for segmentation")
                else:
                    logger.warning("Denoised audio length mismatch; using raw for segments")
            else:
                logger.info("Denoising disabled; using raw audio for segmentation")
        except Exception as e:
            logger.warning(f"Denoising failed; using raw audio for segments: {e}")

        return {
            "audio": audio_denoised if audio_denoised is not None else audio,
            "sample_rate": sample_rate,
            "duration": len(audio) / sample_rate,
            "denoising_metrics": denoising_metrics,
            "silence_removal_metrics": {"applied": False, "reason": "Disabled for timing preservation"},
        }
    
    def load_existing_transcript(self, transcript_path: str) -> Dict:
        """Load existing transcript from JSON file (seconds or milliseconds supported)."""
        logger.info(f"Loading existing transcript: {transcript_path}")
        
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)
        
        # Detect unit: if values look large (e.g., > 100000), assume milliseconds; otherwise seconds
        utterances_in = transcript_data.get('utterances', []) or []
        if utterances_in:
            first = utterances_in[0]
            start_val = float(first.get('start', 0))
            end_val = float(first.get('end', 0))
            is_ms = (start_val >= 1e4) or (end_val >= 1e4)  # >= 10 seconds in ms
        else:
            is_ms = True
        
        # Convert to AssemblyAI-like format with start/end in milliseconds
        class MockUtterance:
            def __init__(self, text, speaker, start, end, confidence=None):
                self.text = text
                self.speaker = speaker
                if is_ms:
                    self.start = int(round(float(start)))
                    self.end = int(round(float(end)))
                else:
                    self.start = int(round(float(start) * 1000.0))
                    self.end = int(round(float(end) * 1000.0))
                self.confidence = confidence or 0.9
        
        class MockTranscript:
            def __init__(self, utterances_data):
                self.utterances = [MockUtterance(**utt) for utt in utterances_data]
        
        transcript = MockTranscript(utterances_in)
        logger.info(f"Loaded transcript: {len(transcript.utterances)} utterances (units={'ms' if is_ms else 's'})")
        return transcript
    
    def transcribe_audio(self, audio_file: str, transcript_path: str = None) -> Dict:
        """Transcribe audio using AssemblyAI or load existing transcript"""
        if transcript_path and os.path.exists(transcript_path):
            return self.load_existing_transcript(transcript_path)
        
        logger.info("Starting transcription...")
        
        # Use the synchronous method
        transcript = self.assemblyai_client.transcribe_audio(audio_file)
        
        if not transcript:
            raise ValueError("Transcription failed")
        
        logger.info(f"Transcription completed: {len(transcript.utterances)} utterances")
        return transcript
    
    def create_segments(self, transcript: Dict, dataset_name: str, audio_data: Dict, source_file: str = None, target_segments_dir: Path = None) -> List[Dict]:
        """Create audio segments from transcript"""
        logger.info("Creating audio segments...")
        
        segments = []
        utterances = transcript.utterances
        # Use the processed audio (denoised if available, otherwise raw)
        audio = audio_data["audio"]
        sample_rate = audio_data["sample_rate"]
        source_stem = Path(source_file).stem if source_file else "source"
        
        # Target directory for segment files (default to dataset/audio)
        segments_dir = target_segments_dir or (Path("dataset") / "audio")
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        for i, utterance in enumerate(utterances):
            # Extract audio segment with exact sample counts (ms -> samples), guard bounds
            start_sample = max(0, int(round(utterance.start * sample_rate / 1000.0)))
            end_sample = max(start_sample, int(round(utterance.end * sample_rate / 1000.0)))
            end_sample = min(len(audio), end_sample)
            num_samples = end_sample - start_sample
            if num_samples <= 0:
                logger.warning(f"Skipping zero-length segment at index {i}: start={start_sample}, end={end_sample}")
                continue
            segment_audio = audio[start_sample:end_sample].astype(np.float32, copy=False)
            
            # Save audio file with source-aware, collision-proof name
            audio_filename = f"{source_stem}_segment_{i}.wav"
            audio_path = segments_dir / audio_filename
            # Write WAV safely
            sf.write(audio_path, segment_audio, sample_rate, subtype='PCM_16')
            
            # Create segment metadata
            segment = {
                "segment_id": f"{source_stem}_segment_{i}",
                "text": utterance.text,
                "speaker_id": utterance.speaker,
                "start_time": utterance.start / 1000.0,  # Convert ms to seconds
                "end_time": utterance.end / 1000.0,
                "duration": (end_sample - start_sample) / sample_rate,
                "confidence": utterance.confidence,
                "audio_file": f"audio/{audio_filename}",
                "source_file": source_file,  # Include original file path for tracing
                "sample_rate": sample_rate  # Preserve original sample rate
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
                "source_file": segment.get("source_file"),  # Include source file for tracing
                "duration": segment["duration"],
                "sample_rate": segment.get("sample_rate", 44100),  # Use actual sample rate
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
        
        # Copy audio files into export/audio directory
        segments_source_dir = self.output_dir / "audio_segments"
        for segment in segments:
            source_audio_path = segments_source_dir / Path(segment["audio_file"]).name
            target_audio_path = audio_dir / Path(segment["audio_file"]).name

            if source_audio_path.exists():
                shutil.copy2(source_audio_path, target_audio_path)
            else:
                logger.warning(f"Missing audio file for export: {source_audio_path}")
        
        logger.info(f"Dataset exported to: {export_dir}")
        return str(export_dir)
    
    def process_dataset(self, sources: List, dataset_name: str) -> Dict[str, Any]:
        """Process complete dataset"""
        logger.info(f"Processing dataset '{dataset_name}' with {len(sources)} sources")
        
        start_time = time.time()
        all_segments = []

        # Canonical dataset directory (single source of truth)
        export_dir = Path("dataset")
        export_dir.mkdir(exist_ok=True)
        export_audio_dir = export_dir / "audio"
        export_audio_dir.mkdir(exist_ok=True)
        export_metadata_path = export_dir / "metadata.json"
        if not export_metadata_path.exists():
            with open(export_metadata_path, 'w') as f:
                json.dump([], f)
        # Transcripts under dataset as well
        export_transcripts_dir = export_dir / "transcripts"
        export_transcripts_dir.mkdir(exist_ok=True)
        
        for i, source in enumerate(sources):
            # Handle both string paths and dict configs
            if isinstance(source, str):
                file_path = source
                transcript_path = None
            else:
                file_path = source.get("file")
                transcript_path = source.get("transcript")
            
            logger.info(f"Processing file {i+1}/{len(sources)}: {file_path}")
            
            try:
                # Process audio
                audio_data = self.process_audio_file(file_path)
                
                # Transcribe (use existing transcript if provided)
                transcript = self.transcribe_audio(file_path, transcript_path)
                
                # Persist per-file transcript (utterances only) under dataset/transcripts
                try:
                    transcript_items = []
                    for utt in getattr(transcript, 'utterances', []) or []:
                        transcript_items.append({
                            "text": utt.text,
                            "speaker": utt.speaker,
                            "start": utt.start,
                            "end": utt.end,
                            "confidence": getattr(utt, 'confidence', None),
                        })
                    stem = Path(file_path).stem
                    transcript_path = export_transcripts_dir / f"{stem}.json"
                    with open(transcript_path, 'w') as tf:
                        json.dump({
                            "source_file": file_path,
                            "sample_rate": audio_data.get("sample_rate"),
                            "utterances": transcript_items
                        }, tf, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to save transcript for {file_path}: {e}")
                
                # Create segments directly under dataset/audio
                segments = self.create_segments(
                    transcript,
                    dataset_name,
                    audio_data,
                    file_path,
                    target_segments_dir=export_audio_dir,
                )
                
                # Validate segments
                valid_segments = self.validate_segments(segments)
                
                all_segments.extend(valid_segments)

                # Append metadata (audio already written to dataset/audio)

                # Append to export metadata.json atomically
                try:
                    with open(export_metadata_path, 'r') as f:
                        current = json.load(f)
                    current.extend([
                        {
                            "segment_id": s["segment_id"],
                            "audio_file": s["audio_file"],
                            "text": s["text"],
                            "speaker_id": s["speaker_id"],
                            "source_file": s.get("source_file"),
                            "duration": s["duration"],
                            "sample_rate": s.get("sample_rate", 44100),
                            "language": "en",
                            "quality_score": s["confidence"],
                            "confidence_score": s["confidence"],
                            "start_time": s["start_time"],
                            "end_time": s["end_time"],
                            "metadata": {"speaker_confidence": s["confidence"]},
                        }
                        for s in valid_segments
                    ])
                    with open(export_metadata_path, 'w') as f:
                        json.dump(current, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed incremental metadata append: {e}")
                
                logger.info(f"✅ File {i+1} completed: {len(valid_segments)} segments")
                
            except Exception as e:
                logger.error(f"❌ File {i+1} failed: {e}")
        
        if not all_segments:
            raise ValueError("No valid segments created")
        
        # No secondary export or mirroring: dataset/ is the canonical output
        export_path = str(export_dir)
        
        # Calculate statistics
        total_duration = sum(s["duration"] for s in all_segments)
        unique_speakers = len(set(s["speaker_id"] for s in all_segments))
        
        processing_time = time.time() - start_time
        
        result = {
            "dataset_name": dataset_name,
            "total_files": len(sources),
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
