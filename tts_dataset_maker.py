#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Dataset Maker - YouTube Video Processor

This script downloads YouTube videos and processes them with AssemblyAI
to create TTS datasets with speaker diarization and transcription.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

def install_dependencies():
    """Install required dependencies."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp", "assemblyai", "-q"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def download_youtube_video(url, output_path="output/audio.wav"):
    """Download YouTube video and extract audio."""
    try:
        # Create output directory
        Path(output_path).parent.mkdir(exist_ok=True)
        
        # Download with yt-dlp
        cmd = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "wav",
            "--audio-quality", "0",  # Best quality
            "-o", output_path,
            url
        ]
        
        print(f"📥 Downloading: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Audio downloaded to: {output_path}")
            return True
        else:
            print(f"✗ Download failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading video: {e}")
        return False

def process_with_assemblyai(audio_path, api_key):
    """Process audio with AssemblyAI for transcription and speaker diarization."""
    try:
        import assemblyai as aai
        
        # Configure AssemblyAI
        aai.settings.api_key = api_key
        
        # Create transcript
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speaker_count=2,  # Adjust based on your needs
            auto_highlights=True,
            auto_chapters=True
        )
        
        print("🎤 Processing with AssemblyAI...")
        transcript = aai.Transcriber().transcribe(audio_path, config)
        
        if transcript.status == aai.TranscriptStatus.error:
            print(f"✗ Transcription failed: {transcript.error}")
            return False
        
        # Save results
        output_path = "output/tts_dataset.json"
        Path(output_path).parent.mkdir(exist_ok=True)
        
        # Convert to our format
        segments = []
        for utterance in transcript.utterances:
            segment = {
                "start": utterance.start / 1000.0,  # Convert to seconds
                "end": utterance.end / 1000.0,
                "speaker": f"Speaker {utterance.speaker}",
                "text": utterance.text
            }
            segments.append(segment)
        
        # Save to JSON
        data = {
            "audio_file": audio_path,
            "segments": segments,
            "metadata": {
                "total_duration": transcript.audio_duration / 1000.0,
                "sample_rate": 16000,  # AssemblyAI standard
                "language": transcript.language_code,
                "created_at": transcript.created_at
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to: {output_path}")
        print(f"✓ Processed {len(segments)} segments")
        return True
        
    except ImportError:
        print("✗ AssemblyAI not installed. Run: pip install assemblyai")
        return False
    except Exception as e:
        print(f"✗ Error processing with AssemblyAI: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Process YouTube video for TTS dataset")
    parser.add_argument("url", help="YouTube URL to process")
    parser.add_argument("--assemblyai-key", required=True, help="AssemblyAI API key")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    print("🎵 TTS Dataset Maker - YouTube Processor")
    print("=" * 50)
    
    # Install dependencies
    print("Installing dependencies...")
    if not install_dependencies():
        print("Failed to install dependencies. Exiting.")
        return 1
    
    # Download video
    audio_path = os.path.join(args.output_dir, "audio.wav")
    if not download_youtube_video(args.url, audio_path):
        print("Failed to download video. Exiting.")
        return 1
    
    # Process with AssemblyAI
    if not process_with_assemblyai(audio_path, args.assemblyai_key):
        print("Failed to process with AssemblyAI. Exiting.")
        return 1
    
    print("\n✅ Processing completed successfully!")
    print(f"📁 Output files:")
    print(f"   - {audio_path}")
    print(f"   - {args.output_dir}/tts_dataset.json")
    print("\n🎯 Next steps:")
    print("   1. Run: python main.py (to explore the dataset)")
    print("   2. Run: python metadata_generator.py (to create training data)")
    
    return 0

if __name__ == "__main__":
    exit(main())

