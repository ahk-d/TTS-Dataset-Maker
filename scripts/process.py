#!/usr/bin/env python3
"""
Simple script to process TTS datasets using the local processor
"""
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tts_pipeline.config.settings import settings
from processor import Processor

def main():
    parser = argparse.ArgumentParser(description="Process TTS Dataset")
    parser.add_argument("config", help="JSON configuration file")
    parser.add_argument("--output", help="Output directory", default="output")
    args = parser.parse_args()

    # Process the dataset
    processor = Processor()
    processor.process_config_file(args.config)

if __name__ == "__main__":
    main()
