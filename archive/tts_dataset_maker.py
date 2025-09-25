#!/usr/bin/env python3
"""
TTS Dataset Maker - Multi-file First Design
Simple wrapper for the TTS dataset creation pipeline
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from tts_service import TTSDatasetService

def process_files(audio_files: List[str], 
                 output_dir: str = "./output",
                 dataset_name: Optional[str] = None,
                 language: str = "en",
                 interactive: bool = True) -> str:
    """
    Process multiple audio files into a TTS dataset
    
    Args:
        audio_files: List of audio file paths
        output_dir: Output directory for the dataset
        dataset_name: Name for the dataset (auto-generated if None)
        language: Language code for transcription
        interactive: Whether to use interactive speaker mapping
        
    Returns:
        Path to the created dataset
    """
    
    print(f"🎵 TTS Dataset Maker - Processing {len(audio_files)} files")
    print("="*60)
    
    # Validate files exist
    missing_files = [f for f in audio_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return None
    
    # Auto-generate dataset name if not provided
    if dataset_name is None:
        dataset_name = f"tts_dataset_{len(audio_files)}_files"
    
    try:
        # Initialize service
        service = TTSDatasetService(output_dir)
        
        # Run complete pipeline
        dataset_metadata, processing_stats = service.run_complete_pipeline(
            audio_files, dataset_name, language
        )
        
        print(f"\n🎉 Dataset created successfully!")
        print(f"📂 Location: {output_dir}")
        print(f"📊 Name: {dataset_metadata.dataset_name}")
        print(f"🎵 Segments: {dataset_metadata.total_segments}")
        print(f"⏱️  Duration: {dataset_metadata.total_duration:.1f}s")
        print(f"👥 Speakers: {len(dataset_metadata.speakers)}")
        
        return output_dir
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TTS Dataset Maker - Multi-file Processing")
    parser.add_argument("files", nargs="+", help="Audio files to process")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--name", "-n", help="Dataset name")
    parser.add_argument("--language", "-l", default="en", help="Language code")
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive mode")
    
    args = parser.parse_args()
    
    # Process files
    result = process_files(
        args.files,
        args.output,
        args.name,
        args.language,
        not args.no_interactive
    )
    
    if result:
        print(f"\n✅ Success! Dataset saved to: {result}")
        return 0
    else:
        print("\n❌ Failed to create dataset")
        return 1

if __name__ == "__main__":
    sys.exit(main())
