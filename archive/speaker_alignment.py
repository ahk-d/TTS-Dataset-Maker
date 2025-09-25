#!/usr/bin/env python3
"""
Speaker Alignment System
Manages speaker ID alignment across multiple files for consistent speaker identification
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from production_storage import ProductionStorage, SourceFile, AudioSegment


@dataclass
class SpeakerMapping:
    """Represents a speaker mapping between files"""
    file_id: str
    original_speaker_id: str
    aligned_speaker_id: str
    confidence: float = 1.0
    notes: str = ""


@dataclass
class SpeakerAlignment:
    """Manages speaker alignments across files"""
    alignment_id: str
    created_at: datetime
    mappings: List[SpeakerMapping]
    global_speakers: Set[str]
    status: str = "draft"  # draft, applied, verified


class SpeakerAlignmentManager:
    """Manages speaker ID alignment across multiple files"""
    
    def __init__(self, dataset_dir: str = "./output"):
        self.dataset_dir = Path(dataset_dir)
        self.storage = ProductionStorage(dataset_dir)
        self.alignments_dir = self.dataset_dir / "speaker_alignments"
        self.alignments_dir.mkdir(exist_ok=True)
    
    def analyze_speakers(self) -> Dict[str, Dict[str, List[str]]]:
        """Analyze speakers across all files and return mapping suggestions"""
        
        # Get all source files
        source_files = self.storage.get_source_files("completed")
        
        # Group speakers by file
        file_speakers = {}
        for file_info in source_files:
            segments = self.storage.get_audio_segments(source_file_id=file_info.file_id)
            speakers = list(set(seg.speaker_id for seg in segments))
            file_speakers[file_info.file_id] = {
                "file_name": Path(file_info.original_path).name,
                "speakers": speakers,
                "segment_count": len(segments)
            }
        
        # Analyze potential alignments
        all_speakers = set()
        for file_data in file_speakers.values():
            all_speakers.update(file_data["speakers"])
        
        # Create alignment suggestions
        alignment_suggestions = {}
        for speaker in all_speakers:
            files_with_speaker = [
                file_id for file_id, data in file_speakers.items()
                if speaker in data["speakers"]
            ]
            alignment_suggestions[speaker] = {
                "files": files_with_speaker,
                "file_names": [file_speakers[fid]["file_name"] for fid in files_with_speaker],
                "total_segments": sum(
                    len([seg for seg in self.storage.get_audio_segments(source_file_id=fid) 
                         if seg.speaker_id == speaker])
                    for fid in files_with_speaker
                )
            }
        
        return {
            "file_speakers": file_speakers,
            "alignment_suggestions": alignment_suggestions,
            "total_unique_speakers": len(all_speakers),
            "total_files": len(source_files)
        }
    
    def create_alignment_template(self, output_file: str = None) -> str:
        """Create a speaker alignment template file"""
        
        if output_file is None:
            output_file = self.alignments_dir / f"speaker_alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Analyze current speakers
        analysis = self.analyze_speakers()
        
        # Create template
        template = {
            "alignment_info": {
                "created_at": datetime.now().isoformat(),
                "total_files": analysis["total_files"],
                "total_unique_speakers": analysis["total_unique_speakers"],
                "status": "draft"
            },
            "file_analysis": analysis["file_speakers"],
            "alignment_suggestions": analysis["alignment_suggestions"],
            "manual_alignments": {},
            "global_speaker_mapping": {},
            "notes": "Edit this file to align speakers across files. Use 'manual_alignments' to map speakers."
        }
        
        # Save template
        with open(output_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Speaker alignment template created: {output_file}")
        return str(output_file)
    
    def load_alignment(self, alignment_file: str) -> SpeakerAlignment:
        """Load speaker alignment from file"""
        
        with open(alignment_file, 'r') as f:
            data = json.load(f)
        
        mappings = []
        for mapping_data in data.get("manual_alignments", {}).get("mappings", []):
            mappings.append(SpeakerMapping(**mapping_data))
        
        return SpeakerAlignment(
            alignment_id=data["alignment_info"]["created_at"],
            created_at=datetime.fromisoformat(data["alignment_info"]["created_at"]),
            mappings=mappings,
            global_speakers=set(data.get("global_speaker_mapping", {}).keys()),
            status=data["alignment_info"]["status"]
        )
    
    def apply_alignment(self, alignment_file: str, dry_run: bool = True) -> Dict[str, Any]:
        """Apply speaker alignment to the dataset"""
        
        alignment = self.load_alignment(alignment_file)
        
        if alignment.status != "draft":
            raise ValueError("Can only apply draft alignments")
        
        # Group mappings by file
        file_mappings = {}
        for mapping in alignment.mappings:
            if mapping.file_id not in file_mappings:
                file_mappings[mapping.file_id] = {}
            file_mappings[mapping.file_id][mapping.original_speaker_id] = mapping.aligned_speaker_id
        
        results = {
            "files_processed": 0,
            "segments_updated": 0,
            "errors": []
        }
        
        if dry_run:
            print("DRY RUN - No changes will be made")
        
        # Apply mappings to each file
        for file_id, mappings in file_mappings.items():
            try:
                # Get segments for this file
                segments = self.storage.get_audio_segments(source_file_id=file_id)
                
                updated_count = 0
                for segment in segments:
                    if segment.speaker_id in mappings:
                        new_speaker_id = mappings[segment.speaker_id]
                        
                        if not dry_run:
                            # Update in database
                            with self.storage.db_path.open() as conn:
                                conn.execute("""
                                    UPDATE audio_segments 
                                    SET speaker_id = ?, metadata = ?
                                    WHERE segment_id = ?
                                """, (
                                    new_speaker_id,
                                    json.dumps({
                                        **json.loads(segment.metadata or "{}"),
                                        "original_speaker_id": segment.speaker_id,
                                        "alignment_applied_at": datetime.now().isoformat()
                                    }),
                                    segment.segment_id
                                ))
                                conn.commit()
                        
                        updated_count += 1
                
                results["files_processed"] += 1
                results["segments_updated"] += updated_count
                
                print(f"File {file_id}: {updated_count} segments updated")
                
            except Exception as e:
                error_msg = f"Error processing file {file_id}: {e}"
                results["errors"].append(error_msg)
                print(error_msg)
        
        if not dry_run:
            # Mark alignment as applied
            alignment.status = "applied"
            self._save_alignment(alignment_file, alignment)
        
        return results
    
    def verify_alignment(self, alignment_file: str) -> Dict[str, Any]:
        """Verify speaker alignment results"""
        
        alignment = self.load_alignment(alignment_file)
        
        # Get current speaker distribution
        all_segments = self.storage.get_audio_segments()
        current_speakers = {}
        
        for segment in all_segments:
            if segment.speaker_id not in current_speakers:
                current_speakers[segment.speaker_id] = {
                    "total_segments": 0,
                    "total_duration": 0.0,
                    "files": set()
                }
            
            current_speakers[segment.speaker_id]["total_segments"] += 1
            current_speakers[segment.speaker_id]["total_duration"] += segment.duration
            current_speakers[segment.speaker_id]["files"].add(segment.source_file_id)
        
        # Convert sets to lists for JSON serialization
        for speaker_data in current_speakers.values():
            speaker_data["files"] = list(speaker_data["files"])
        
        return {
            "alignment_status": alignment.status,
            "total_speakers": len(current_speakers),
            "speaker_distribution": current_speakers,
            "alignment_mappings": len(alignment.mappings)
        }
    
    def interactive_alignment(self) -> str:
        """Interactive speaker alignment tool with manual segment review"""
        
        print("🎤 Speaker Alignment Tool")
        print("=" * 50)
        
        # Analyze current speakers
        analysis = self.analyze_speakers()
        
        print(f"Found {analysis['total_files']} files with {analysis['total_unique_speakers']} unique speakers")
        print("\nFile Analysis:")
        for file_id, data in analysis["file_speakers"].items():
            print(f"  📁 {data['file_name']}: {data['speakers']} ({data['segment_count']} segments)")
        
        # Create alignment template
        alignment_file = self.create_alignment_template()
        
        print(f"\n📝 Alignment template created: {alignment_file}")
        print("\n🔍 Manual Alignment Process:")
        print("1. Review segments for each file")
        print("2. Identify which segments belong to the same speaker")
        print("3. Edit the alignment file to map speakers")
        print("4. Test with dry run: python dataset_manager.py apply-alignment <file> --dry-run")
        print("5. Apply alignment: python dataset_manager.py apply-alignment <file>")
        
        return alignment_file
    
    def review_segments_for_file(self, file_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample segments from a file for manual review"""
        
        segments = self.storage.get_audio_segments(source_file_id=file_id)
        
        # Group by speaker
        speaker_segments = {}
        for segment in segments:
            if segment.speaker_id not in speaker_segments:
                speaker_segments[segment.speaker_id] = []
            speaker_segments[segment.speaker_id].append(segment)
        
        # Get sample segments for each speaker
        review_data = []
        for speaker_id, segs in speaker_segments.items():
            # Get a few sample segments
            sample_segments = segs[:limit]
            
            review_data.append({
                "speaker_id": speaker_id,
                "total_segments": len(segs),
                "sample_segments": [
                    {
                        "segment_id": seg.segment_id,
                        "text": seg.text,
                        "duration": seg.duration,
                        "audio_file": seg.audio_file_path,
                        "quality_score": seg.quality_score
                    }
                    for seg in sample_segments
                ]
            })
        
        return review_data
    
    def create_alignment_workflow(self) -> str:
        """Create a step-by-step alignment workflow"""
        
        print("🎯 Speaker Alignment Workflow")
        print("=" * 50)
        
        # Get all files
        source_files = self.storage.get_source_files("completed")
        
        if not source_files:
            print("❌ No completed files found. Process some files first.")
            return None
        
        print(f"📁 Found {len(source_files)} completed files")
        
        # Create workflow file
        workflow_file = self.alignments_dir / f"alignment_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        workflow = {
            "workflow_info": {
                "created_at": datetime.now().isoformat(),
                "total_files": len(source_files),
                "status": "draft"
            },
            "files": [],
            "speaker_alignments": {},
            "notes": "Follow the workflow to align speakers across files"
        }
        
        # Process each file
        for file_info in source_files:
            print(f"\n📁 Processing: {Path(file_info.original_path).name}")
            
            # Get sample segments
            review_data = self.review_segments_for_file(file_info.file_id)
            
            file_workflow = {
                "file_id": file_info.file_id,
                "file_name": Path(file_info.original_path).name,
                "speakers": [data["speaker_id"] for data in review_data],
                "review_data": review_data,
                "alignment_notes": "",
                "aligned_speakers": {}
            }
            
            workflow["files"].append(file_workflow)
            
            print(f"  🎤 Speakers: {[data['speaker_id'] for data in review_data]}")
            print(f"  📊 Total segments: {sum(data['total_segments'] for data in review_data)}")
        
        # Save workflow
        with open(workflow_file, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        print(f"\n📝 Workflow created: {workflow_file}")
        print("\n🔍 Next Steps:")
        print("1. Review the workflow file")
        print("2. For each file, identify which speakers are the same person")
        print("3. Add alignment mappings to 'speaker_alignments' section")
        print("4. Test with dry run: python dataset_manager.py apply-alignment <file> --dry-run")
        print("5. Apply alignment: python dataset_manager.py apply-alignment <file>")
        
        return str(workflow_file)
    
    def create_speaker_reference(self) -> str:
        """Create a speaker reference guide with sample segments"""
        
        print("🎤 Creating Speaker Reference Guide")
        print("=" * 50)
        
        # Get all segments grouped by speaker
        all_segments = self.storage.get_audio_segments()
        speaker_groups = {}
        
        for segment in all_segments:
            if segment.speaker_id not in speaker_groups:
                speaker_groups[segment.speaker_id] = []
            speaker_groups[segment.speaker_id].append(segment)
        
        # Create reference file
        reference_file = self.alignments_dir / f"speaker_reference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        reference = {
            "reference_info": {
                "created_at": datetime.now().isoformat(),
                "total_speakers": len(speaker_groups),
                "total_segments": len(all_segments)
            },
            "speakers": {}
        }
        
        for speaker_id, segments in speaker_groups.items():
            # Get sample segments (first 3)
            sample_segments = segments[:3]
            
            reference["speakers"][speaker_id] = {
                "total_segments": len(segments),
                "total_duration": sum(seg.duration for seg in segments),
                "files": list(set(seg.source_file_id for seg in segments)),
                "sample_segments": [
                    {
                        "segment_id": seg.segment_id,
                        "text": seg.text,
                        "duration": seg.duration,
                        "audio_file": seg.audio_file_path,
                        "source_file_id": seg.source_file_id,
                        "quality_score": seg.quality_score
                    }
                    for seg in sample_segments
                ]
            }
        
        # Save reference
        with open(reference_file, 'w') as f:
            json.dump(reference, f, indent=2)
        
        print(f"📝 Speaker reference created: {reference_file}")
        print(f"🎤 Found {len(speaker_groups)} unique speakers")
        
        # Print summary
        for speaker_id, data in reference["speakers"].items():
            print(f"  🎤 Speaker {speaker_id}: {data['total_segments']} segments, {data['total_duration']:.1f}s")
            print(f"    📁 Files: {data['files']}")
            print(f"    📝 Sample: '{data['sample_segments'][0]['text'][:50]}...'")
        
        return str(reference_file)
    
    def _save_alignment(self, alignment_file: str, alignment: SpeakerAlignment):
        """Save alignment to file"""
        
        data = {
            "alignment_info": {
                "created_at": alignment.created_at.isoformat(),
                "status": alignment.status
            },
            "manual_alignments": {
                "mappings": [asdict(mapping) for mapping in alignment.mappings]
            },
            "global_speaker_mapping": {speaker: speaker for speaker in alignment.global_speakers}
        }
        
        with open(alignment_file, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    """Main CLI interface for speaker alignment"""
    parser = argparse.ArgumentParser(description="Speaker Alignment Tool")
    parser.add_argument("--dataset-dir", "-d", default="./output", help="Dataset directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    subparsers.add_parser("analyze", help="Analyze speakers across files")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Interactive alignment tool")
    
    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Apply speaker alignment")
    apply_parser.add_argument("alignment_file", help="Alignment file to apply")
    apply_parser.add_argument("--dry-run", action="store_true", help="Dry run (no changes)")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify alignment results")
    verify_parser.add_argument("alignment_file", help="Alignment file to verify")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize alignment manager
    manager = SpeakerAlignmentManager(args.dataset_dir)
    
    try:
        if args.command == "analyze":
            analysis = manager.analyze_speakers()
            print(json.dumps(analysis, indent=2))
        
        elif args.command == "interactive":
            alignment_file = manager.interactive_alignment()
            print(f"\n✅ Alignment template created: {alignment_file}")
        
        elif args.command == "apply":
            results = manager.apply_alignment(args.alignment_file, args.dry_run)
            print(f"\n📊 Results:")
            print(f"  Files processed: {results['files_processed']}")
            print(f"  Segments updated: {results['segments_updated']}")
            if results['errors']:
                print(f"  Errors: {len(results['errors'])}")
                for error in results['errors']:
                    print(f"    ❌ {error}")
        
        elif args.command == "verify":
            verification = manager.verify_alignment(args.alignment_file)
            print(json.dumps(verification, indent=2))
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
