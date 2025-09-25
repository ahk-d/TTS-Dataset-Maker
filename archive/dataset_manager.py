#!/usr/bin/env python3
"""
Dataset Manager - Production-ready CLI for TTS Dataset Management
Provides comprehensive dataset management, statistics, and export capabilities
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tts_service import TTSDatasetService
from production_storage import ProductionStorage
from speaker_alignment import SpeakerAlignmentManager


class DatasetManager:
    """Production-ready dataset management CLI"""
    
    def __init__(self, dataset_dir: str = "./output"):
        self.dataset_dir = dataset_dir
        self.service = TTSDatasetService(dataset_dir)
        self.storage = ProductionStorage(dataset_dir)
        self.alignment_manager = SpeakerAlignmentManager(dataset_dir)
    
    def status(self) -> Dict[str, Any]:
        """Get comprehensive dataset status"""
        stats = self.storage.get_dataset_statistics()
        source_files = self.storage.get_source_files()
        
        return {
            "dataset_directory": self.dataset_dir,
            "database_path": str(self.storage.db_path),
            "statistics": stats,
            "recent_files": [
                {
                    "file_id": f.file_id,
                    "original_path": f.original_path,
                    "status": f.status,
                    "duration": f.duration,
                    "created_at": f.created_at.isoformat()
                }
                for f in source_files[:10]  # Last 10 files
            ]
        }
    
    def list_files(self, status: str = None) -> List[Dict[str, Any]]:
        """List source files with optional status filter"""
        files = self.storage.get_source_files(status)
        
        return [
            {
                "file_id": f.file_id,
                "original_path": f.original_path,
                "file_type": f.file_type,
                "file_size_mb": round(f.file_size / (1024 * 1024), 2),
                "duration_seconds": f.duration,
                "status": f.status,
                "created_at": f.created_at.isoformat(),
                "processed_at": f.processed_at.isoformat() if f.processed_at else None,
                "error_message": f.error_message
            }
            for f in files
        ]
    
    def list_segments(self, source_file_id: str = None, speaker_id: str = None, 
                     limit: int = 100) -> List[Dict[str, Any]]:
        """List audio segments with optional filters"""
        segments = self.storage.get_audio_segments(source_file_id, speaker_id)
        
        # Limit results
        segments = segments[:limit]
        
        return [
            {
                "segment_id": s.segment_id,
                "source_file_id": s.source_file_id,
                "text": s.text[:100] + "..." if len(s.text) > 100 else s.text,
                "speaker_id": s.speaker_id,
                "duration_seconds": s.duration,
                "quality_score": s.quality_score,
                "confidence_score": s.confidence_score,
                "language": s.language,
                "created_at": s.created_at.isoformat()
            }
            for s in segments
        ]
    
    def list_speakers(self) -> List[Dict[str, Any]]:
        """List all speakers with statistics"""
        speakers = self.service.list_speakers()
        
        speaker_stats = []
        for speaker_id in speakers:
            stats = self.service.get_speaker_statistics(speaker_id)
            speaker_stats.append(stats)
        
        return speaker_stats
    
    def export_dataset(self, dataset_name: str, output_format: str = "huggingface") -> str:
        """Export dataset in specified format"""
        export_path = self.service.create_dataset_export(dataset_name, output_format)
        return export_path
    
    def get_file_details(self, file_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific file"""
        files = self.storage.get_source_files()
        file_info = next((f for f in files if f.file_id == file_id), None)
        
        if not file_info:
            return {"error": f"File '{file_id}' not found"}
        
        # Get segments for this file
        segments = self.storage.get_audio_segments(source_file_id=file_id)
        
        return {
            "file_info": {
                "file_id": file_info.file_id,
                "original_path": file_info.original_path,
                "file_type": file_info.file_type,
                "file_size_bytes": file_info.file_size,
                "duration_seconds": file_info.duration,
                "sample_rate": file_info.sample_rate,
                "channels": file_info.channels,
                "format": file_info.format,
                "hash": file_info.hash,
                "status": file_info.status,
                "created_at": file_info.created_at.isoformat(),
                "processed_at": file_info.processed_at.isoformat() if file_info.processed_at else None,
                "error_message": file_info.error_message,
                "metadata": file_info.metadata
            },
            "segments": {
                "total_count": len(segments),
                "total_duration_seconds": sum(s.duration for s in segments),
                "speakers": list(set(s.speaker_id for s in segments)),
                "average_quality": sum(s.quality_score for s in segments) / len(segments) if segments else 0,
                "average_confidence": sum(s.confidence_score for s in segments) / len(segments) if segments else 0
            }
        }
    
    def cleanup_failed_files(self) -> int:
        """Remove failed files and their associated data"""
        failed_files = self.storage.get_source_files("failed")
        count = 0
        
        for file_info in failed_files:
            # Remove from database
            with self.storage.db_path.open() as conn:
                conn.execute("DELETE FROM source_files WHERE file_id = ?", (file_info.file_id,))
                conn.execute("DELETE FROM audio_segments WHERE source_file_id = ?", (file_info.file_id,))
                conn.commit()
            
            # Remove stored file
            stored_file = self.storage.files_dir / f"{file_info.file_id}.{file_info.format}"
            if stored_file.exists():
                stored_file.unlink()
            
            count += 1
        
        return count
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate dataset integrity"""
        issues = []
        
        # Check source files
        source_files = self.storage.get_source_files()
        for file_info in source_files:
            stored_file = self.storage.files_dir / f"{file_info.file_id}.{file_info.format}"
            if not stored_file.exists():
                issues.append(f"Missing stored file for {file_info.file_id}")
        
        # Check audio segments
        segments = self.storage.get_audio_segments()
        for segment in segments:
            if not Path(segment.audio_file_path).exists():
                issues.append(f"Missing audio file for segment {segment.segment_id}")
        
        # Check database integrity
        with self.storage.db_path.open() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM audio_segments 
                WHERE source_file_id NOT IN (SELECT file_id FROM source_files)
            """)
            orphaned_segments = cursor.fetchone()[0]
            if orphaned_segments > 0:
                issues.append(f"Found {orphaned_segments} orphaned segments")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_issues": len(issues)
        }


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="TTS Dataset Manager - Production CLI")
    parser.add_argument("--dataset-dir", "-d", default="./output", help="Dataset directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    subparsers.add_parser("status", help="Show dataset status")
    
    # List files command
    files_parser = subparsers.add_parser("list-files", help="List source files")
    files_parser.add_argument("--status", choices=["pending", "processing", "completed", "failed"], 
                            help="Filter by status")
    
    # List segments command
    segments_parser = subparsers.add_parser("list-segments", help="List audio segments")
    segments_parser.add_argument("--source-file-id", help="Filter by source file ID")
    segments_parser.add_argument("--speaker-id", help="Filter by speaker ID")
    segments_parser.add_argument("--limit", type=int, default=100, help="Limit results")
    
    # List speakers command
    subparsers.add_parser("list-speakers", help="List all speakers")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export dataset")
    export_parser.add_argument("dataset_name", help="Dataset name")
    export_parser.add_argument("--format", choices=["huggingface", "orpheus"], 
                             default="huggingface", help="Export format")
    
    # File details command
    details_parser = subparsers.add_parser("file-details", help="Get file details")
    details_parser.add_argument("file_id", help="File ID")
    
    # Cleanup command
    subparsers.add_parser("cleanup", help="Clean up failed files")
    
    # Validate command
    subparsers.add_parser("validate", help="Validate dataset integrity")
    
    # Speaker alignment commands
    subparsers.add_parser("analyze-speakers", help="Analyze speakers across files")
    subparsers.add_parser("align-speakers", help="Interactive speaker alignment")
    subparsers.add_parser("create-workflow", help="Create alignment workflow")
    subparsers.add_parser("create-reference", help="Create speaker reference guide")
    
    # Apply alignment command
    apply_parser = subparsers.add_parser("apply-alignment", help="Apply speaker alignment")
    apply_parser.add_argument("alignment_file", help="Alignment file to apply")
    apply_parser.add_argument("--dry-run", action="store_true", help="Dry run (no changes)")
    
    # Verify alignment command
    verify_parser = subparsers.add_parser("verify-alignment", help="Verify alignment results")
    verify_parser.add_argument("alignment_file", help="Alignment file to verify")
    
    # Web UI command
    ui_parser = subparsers.add_parser("ui", help="Start web annotation UI")
    ui_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    ui_parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    ui_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize dataset manager
    manager = DatasetManager(args.dataset_dir)
    
    try:
        if args.command == "status":
            result = manager.status()
            print(json.dumps(result, indent=2))
        
        elif args.command == "list-files":
            result = manager.list_files(args.status)
            print(json.dumps(result, indent=2))
        
        elif args.command == "list-segments":
            result = manager.list_segments(args.source_file_id, args.speaker_id, args.limit)
            print(json.dumps(result, indent=2))
        
        elif args.command == "list-speakers":
            result = manager.list_speakers()
            print(json.dumps(result, indent=2))
        
        elif args.command == "export":
            export_path = manager.export_dataset(args.dataset_name, args.format)
            print(f"Dataset exported to: {export_path}")
        
        elif args.command == "file-details":
            result = manager.get_file_details(args.file_id)
            print(json.dumps(result, indent=2))
        
        elif args.command == "cleanup":
            count = manager.cleanup_failed_files()
            print(f"Cleaned up {count} failed files")
        
        elif args.command == "validate":
            result = manager.validate_dataset()
            print(json.dumps(result, indent=2))
            
            if not result["valid"]:
                return 1
        
        elif args.command == "analyze-speakers":
            analysis = manager.alignment_manager.analyze_speakers()
            print(json.dumps(analysis, indent=2))
        
        elif args.command == "align-speakers":
            alignment_file = manager.alignment_manager.interactive_alignment()
            print(f"\n✅ Alignment template created: {alignment_file}")
        
        elif args.command == "create-workflow":
            workflow_file = manager.alignment_manager.create_alignment_workflow()
            if workflow_file:
                print(f"\n✅ Workflow created: {workflow_file}")
            else:
                print("\n❌ No completed files found")
        
        elif args.command == "create-reference":
            reference_file = manager.alignment_manager.create_speaker_reference()
            print(f"\n✅ Speaker reference created: {reference_file}")
        
        elif args.command == "apply-alignment":
            results = manager.alignment_manager.apply_alignment(args.alignment_file, args.dry_run)
            print(f"\n📊 Results:")
            print(f"  Files processed: {results['files_processed']}")
            print(f"  Segments updated: {results['segments_updated']}")
            if results['errors']:
                print(f"  Errors: {len(results['errors'])}")
                for error in results['errors']:
                    print(f"    ❌ {error}")
        
        elif args.command == "verify-alignment":
            verification = manager.alignment_manager.verify_alignment(args.alignment_file)
            print(json.dumps(verification, indent=2))
        
        elif args.command == "ui":
            from speech_annotation_ui import SpeechAnnotationUI
            ui = SpeechAnnotationUI(args.dataset_dir, args.host, args.port)
            ui.run(args.debug)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
