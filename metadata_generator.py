#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metadata Generator for TTS Dataset Maker

This script generates metadata files and extracts individual audio segments
for each speaker segment, creating a structured dataset for TTS training.
"""

import json
import os
import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Any

class MetadataGenerator:
    def __init__(self, json_path="output/tts_dataset.json", audio_path="output/audio.wav", output_dir="output/segments"):
        self.json_path = json_path
        self.audio_path = audio_path
        self.output_dir = Path(output_dir)
        self.segments_dir = self.output_dir / "audio"
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.segments_dir.mkdir(exist_ok=True)
        
        # Load data
        self._load_data()
        self._load_audio()
    
    def _load_data(self):
        """Load and process JSON data."""
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        segments = data.get("segments", [])
        if not segments:
            raise ValueError("No segments found in JSON under key 'segments'.")
        
        # Convert to DataFrame
        self.df = pd.DataFrame(segments)[["start", "end", "speaker", "text"]].copy()
        self.df["start"] = self.df["start"].astype(float)
        self.df["end"] = self.df["end"].astype(float)
        self.df["duration_s"] = (self.df["end"] - self.df["start"]).round(3)
        self.df.insert(0, "id", range(len(self.df)))
    
    def _load_audio(self):
        """Load the full audio file."""
        self.full_audio, self.sr = sf.read(self.audio_path, always_2d=False)
        print(f"Loaded audio: {self.full_audio.shape}, Sample rate: {self.sr} Hz")
    
    def extract_audio_segment(self, start_s: float, end_s: float, segment_id: int) -> str:
        """Extract audio segment and save to file."""
        start_idx = int(round(start_s * self.sr))
        end_idx = int(round(end_s * self.sr))
        
        # Extract segment
        audio_segment = self.full_audio[start_idx:end_idx]
        
        # Generate filename
        filename = f"segment_{segment_id:06d}.wav"
        filepath = self.segments_dir / filename
        
        # Save segment
        sf.write(filepath, audio_segment, self.sr)
        
        return str(filepath)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for TTS training."""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Basic punctuation normalization
        text = text.replace("...", "…")
        text = text.replace("--", "—")
        
        # Remove any non-printable characters
        text = "".join(char for char in text if char.isprintable() or char.isspace())
        
        return text.strip()
    
    def generate_metadata(self, min_duration: float = 0.5, max_duration: float = 30.0):
        """Generate metadata files and extract audio segments."""
        print(f"Processing {len(self.df)} segments...")
        
        metadata_raw = []
        metadata_clean = []
        
        for idx, row in self.df.iterrows():
            start_s = float(row["start"])
            end_s = float(row["end"])
            duration_s = float(row["duration_s"])
            speaker = str(row["speaker"])
            text = str(row["text"])
            
            # Filter by duration
            if duration_s < min_duration or duration_s > max_duration:
                print(f"Skipping segment {idx}: duration {duration_s:.2f}s (outside range {min_duration}-{max_duration}s)")
                continue
            
            # Clean text
            clean_text = self.clean_text(text)
            
            # Skip if text is too short after cleaning
            if len(clean_text.strip()) < 3:
                print(f"Skipping segment {idx}: text too short after cleaning")
                continue
            
            # Extract audio segment
            try:
                audio_path = self.extract_audio_segment(start_s, end_s, idx)
                
                # Raw metadata (original data)
                raw_entry = {
                    "id": int(row["id"]),
                    "speaker": speaker,
                    "start": start_s,
                    "end": end_s,
                    "duration_s": duration_s,
                    "text": text,
                    "audio_path": audio_path,
                    "sample_rate": self.sr
                }
                metadata_raw.append(raw_entry)
                
                # Clean metadata (for TTS training)
                clean_entry = {
                    "id": int(row["id"]),
                    "speaker": speaker,
                    "start": start_s,
                    "end": end_s,
                    "duration_s": duration_s,
                    "text": clean_text,
                    "audio_path": audio_path,
                    "sample_rate": self.sr
                }
                metadata_clean.append(clean_entry)
                
                print(f"✓ Segment {idx}: {speaker} ({duration_s:.2f}s) - {clean_text[:50]}...")
                
            except Exception as e:
                print(f"✗ Error processing segment {idx}: {e}")
                continue
        
        # Save metadata files
        self._save_metadata(metadata_raw, "metadata_raw.json")
        self._save_metadata(metadata_clean, "metadata.json")
        
        print(f"\n✅ Generated {len(metadata_raw)} segments")
        print(f"📁 Audio segments saved to: {self.segments_dir}")
        print(f"📄 Metadata files saved to: {self.output_dir}")
        
        # Print statistics
        self._print_statistics(metadata_clean)
    
    def _save_metadata(self, metadata: List[Dict[str, Any]], filename: str):
        """Save metadata to JSON file."""
        filepath = self.output_dir / filename
        
        output_data = {
            "dataset_info": {
                "total_segments": len(metadata),
                "sample_rate": self.sr,
                "source_audio": self.audio_path,
                "generated_at": pd.Timestamp.now().isoformat()
            },
            "segments": metadata
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {filename} with {len(metadata)} segments")
    
    def _print_statistics(self, metadata: List[Dict[str, Any]]):
        """Print dataset statistics."""
        if not metadata:
            return
        
        df_stats = pd.DataFrame(metadata)
        
        print("\n📊 Dataset Statistics:")
        print(f"   Total segments: {len(metadata)}")
        print(f"   Unique speakers: {df_stats['speaker'].nunique()}")
        print(f"   Total duration: {df_stats['duration_s'].sum():.2f}s")
        print(f"   Average duration: {df_stats['duration_s'].mean():.2f}s")
        print(f"   Min duration: {df_stats['duration_s'].min():.2f}s")
        print(f"   Max duration: {df_stats['duration_s'].max():.2f}s")
        
        # Speaker statistics
        speaker_stats = df_stats.groupby('speaker').agg({
            'id': 'count',
            'duration_s': ['sum', 'mean']
        }).round(2)
        
        print(f"\n👥 Speaker Statistics:")
        for speaker in speaker_stats.index:
            count = speaker_stats.loc[speaker, ('id', 'count')]
            total_duration = speaker_stats.loc[speaker, ('duration_s', 'sum')]
            avg_duration = speaker_stats.loc[speaker, ('duration_s', 'mean')]
            print(f"   {speaker}: {count} segments, {total_duration:.2f}s total, {avg_duration:.2f}s avg")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate metadata and extract audio segments")
    parser.add_argument("--json-path", default="output/tts_dataset.json", help="Path to JSON file")
    parser.add_argument("--audio-path", default="output/audio.wav", help="Path to audio file")
    parser.add_argument("--output-dir", default="output/segments", help="Output directory")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Minimum segment duration (seconds)")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Maximum segment duration (seconds)")
    
    args = parser.parse_args()
    
    print("🎵 TTS Dataset Metadata Generator")
    print("=" * 50)
    
    try:
        generator = MetadataGenerator(
            json_path=args.json_path,
            audio_path=args.audio_path,
            output_dir=args.output_dir
        )
        
        generator.generate_metadata(
            min_duration=args.min_duration,
            max_duration=args.max_duration
        )
        
        print("\n✅ Metadata generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 