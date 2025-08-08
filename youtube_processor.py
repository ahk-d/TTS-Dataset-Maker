#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube Processor - Wrapper for TTS Dataset Maker

This script is a wrapper around the main tts_dataset_maker.py script
for processing YouTube videos with AssemblyAI.
"""

import subprocess
import sys
import argparse

def process_youtube_video(url, assemblyai_key):
    """Process a YouTube video with AssemblyAI using the main script."""
    try:
        # Run the main TTS dataset maker script
        cmd = [
            sys.executable, 
            "tts_dataset_maker.py", 
            url, 
            "--assemblyai-key", 
            assemblyai_key
        ]
        
        print(f"🎵 Processing YouTube video: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ YouTube video processed successfully!")
            print("📁 Output files should be in the 'output' directory.")
            return True
        else:
            print(f"❌ Error processing video: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running TTS dataset maker: {e}")
        return False

def main():
    """Main function for YouTube processing."""
    parser = argparse.ArgumentParser(description="Process YouTube video for TTS dataset")
    parser.add_argument("url", help="YouTube URL to process")
    parser.add_argument("--assemblyai-key", required=True, help="AssemblyAI API key")
    
    args = parser.parse_args()
    
    print("🎵 TTS Dataset Maker - YouTube Processor")
    print("=" * 50)
    
    if process_youtube_video(args.url, args.assemblyai_key):
        print("\n✅ Processing completed successfully!")
        print("\n🎯 Next steps:")
        print("   1. Run: python main.py (to explore the dataset)")
        print("   2. Run: python metadata_generator.py (to create training data)")
    else:
        print("\n❌ Processing failed.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 