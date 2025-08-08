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



def download_youtube_video(url, output_path="output/audio.wav", force_download=False):
    """Download YouTube video and extract audio with caching."""
    try:
        # Create output directory
        Path(output_path).parent.mkdir(exist_ok=True)
        
        # Generate cache key based on URL
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        cache_path = output_path.replace('.wav', f'_{url_hash}.wav')
        
        # Check if cached file already exists for this specific URL
        if os.path.exists(cache_path) and not force_download:
            file_size = os.path.getsize(cache_path)
            if file_size > 1024:  # At least 1KB to ensure it's not empty
                print(f"📁 Audio file already exists for this URL: {cache_path}")
                print(f"📊 File size: {file_size / (1024*1024):.2f} MB")
                print("⏭️  Skipping download (use --force-download to re-download)")
                # Copy cached file to requested output path
                import shutil
                shutil.copy2(cache_path, output_path)
                return True
            else:
                print(f"⚠️  Existing cached file is too small ({file_size} bytes), re-downloading...")
        elif force_download and os.path.exists(cache_path):
            print("🔄 Force download requested, re-downloading...")
        
        # Download with yt-dlp to cache path
        cmd = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "wav",
            "--audio-quality", "0",  # Best quality
            "-o", cache_path,
            url
        ]
        
        print(f"📥 Downloading: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Copy from cache to final output path
            import shutil
            shutil.copy2(cache_path, output_path)
            file_size = os.path.getsize(output_path)
            print(f"✓ Audio downloaded to: {output_path}")
            print(f"📊 File size: {file_size / (1024*1024):.2f} MB")
            return True
        else:
            print(f"✗ Download failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading video: {e}")
        return False

def denoise_audio(audio_path, denoised_path=None):
    """Denoise audio using the denoiser module."""
    try:
        from denoiser import Denoiser
        
        if denoised_path is None:
            denoised_path = audio_path.replace('.wav', '_denoised.wav')
        
        print("🔇 Denoising audio...")
        denoiser = Denoiser()
        success = denoiser.denoise_file(audio_path, denoised_path)
        
        if success:
            print(f"✓ Denoised audio saved to: {denoised_path}")
            return denoised_path
        else:
            print("⚠️  Denoising failed, using original audio")
            return audio_path
        
    except ImportError as e:
        print(f"⚠️  Denoiser not available: {e}")
        print("💡 Install DeepFilterNet with: pip install deepfilternet")
        return audio_path
    except Exception as e:
        print(f"⚠️  Denoising failed: {e}, using original audio")
        return audio_path

def process_with_assemblyai(audio_path, api_key, url=None, force_process=False):
    """Process audio with AssemblyAI for transcription and speaker diarization."""
    try:
        import assemblyai as aai
        
        # Configure AssemblyAI
        aai.settings.api_key = api_key
        
        # Generate cache key based on URL
        if url:
            import hashlib
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            json_path = f"output/tts_dataset_{url_hash}.json"
        else:
            json_path = "output/tts_dataset.json"
        
        # Check if JSON already exists for this specific URL
        if os.path.exists(json_path) and not force_process:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                segments = data.get("segments", [])
                if segments:
                    print(f"📁 Transcription already exists for this URL: {json_path}")
                    print(f"📊 Found {len(segments)} segments")
                    print("⏭️  Skipping AssemblyAI processing (use --force-process to re-process)")
                    # Copy cached JSON to standard output path
                    import shutil
                    shutil.copy2(json_path, "output/tts_dataset.json")
                    return True
            except Exception as e:
                print(f"⚠️  Error reading existing JSON: {e}, re-processing...")
        
        # Create transcript
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            auto_highlights=True,
            auto_chapters=True
        )
        
        print("🎤 Processing with AssemblyAI...")
        transcript = aai.Transcriber().transcribe(audio_path, config)
        
        if transcript.status == aai.TranscriptStatus.error:
            print(f"✗ Transcription failed: {transcript.error}")
            return False
        
        # Save results to cache path
        Path(json_path).parent.mkdir(exist_ok=True)
        
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
            }
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Copy to standard output path
        import shutil
        shutil.copy2(json_path, "output/tts_dataset.json")
        
        print(f"✓ Results saved to: {json_path}")
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
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if file exists")
    parser.add_argument("--force-process", action="store_true", help="Force re-process with AssemblyAI even if JSON exists")
    parser.add_argument("--denoise", action="store_true", help="Apply denoising before transcription")
    
    args = parser.parse_args()
    
    print("🎵 TTS Dataset Maker - YouTube Processor")
    print("=" * 50)
    
    # Download video
    audio_path = os.path.join(args.output_dir, "audio.wav")
    if not download_youtube_video(args.url, audio_path, args.force_download):
        print("Failed to download video. Exiting.")
        return 1
    
    # Denoise audio if requested
    if args.denoise:
        denoised_path = os.path.join(args.output_dir, "audio_denoised.wav")
        audio_path = denoise_audio(audio_path, denoised_path)
    
    # Process with AssemblyAI
    if not process_with_assemblyai(audio_path, args.assemblyai_key, args.url, args.force_process):
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

