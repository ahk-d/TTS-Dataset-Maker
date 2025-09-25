#!/usr/bin/env python3
"""
Production-Ready Storage System for TTS Dataset Maker
Intelligent metadata management with source tracking, versioning, and scalability
"""

import os
import json
import shutil
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid

from pydantic import BaseModel, Field
from config import settings


@dataclass
class SourceFile:
    """Represents a source file with metadata"""
    file_id: str
    original_path: str
    file_type: str  # audio, video, youtube
    file_size: int
    duration: float
    sample_rate: int
    channels: int
    format: str  # mp3, wav, mp4, etc.
    hash: str  # file hash for deduplication
    created_at: datetime
    processed_at: Optional[datetime] = None
    status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class AudioSegment:
    """Represents an audio segment with full metadata"""
    segment_id: str
    source_file_id: str
    audio_file_path: str
    text: str
    speaker_id: str
    start_time: float
    end_time: float
    duration: float
    sample_rate: int
    quality_score: float
    confidence_score: float
    language: str
    created_at: datetime
    metadata: Dict[str, Any] = None


class ProductionStorage:
    """Production-ready storage system with intelligent metadata management"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or settings.output_dir)
        self.db_path = self.base_dir / "dataset.db"
        self.files_dir = self.base_dir / "source_files"
        self.segments_dir = self.base_dir / "audio_segments"
        self.metadata_dir = self.base_dir / "metadata"
        self.temp_dir = self.base_dir / "temp"
        
        # Create directories
        for dir_path in [self.files_dir, self.segments_dir, self.metadata_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metadata tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS source_files (
                    file_id TEXT PRIMARY KEY,
                    original_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER,
                    duration REAL,
                    sample_rate INTEGER,
                    channels INTEGER,
                    format TEXT,
                    hash TEXT UNIQUE,
                    created_at TIMESTAMP,
                    processed_at TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audio_segments (
                    segment_id TEXT PRIMARY KEY,
                    source_file_id TEXT NOT NULL,
                    audio_file_path TEXT NOT NULL,
                    text TEXT NOT NULL,
                    speaker_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    duration REAL NOT NULL,
                    sample_rate INTEGER,
                    quality_score REAL,
                    confidence_score REAL,
                    language TEXT,
                    created_at TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (source_file_id) REFERENCES source_files (file_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    total_files INTEGER,
                    processed_files INTEGER,
                    failed_files INTEGER,
                    total_segments INTEGER,
                    status TEXT
                )
            """)
            
            conn.commit()
    
    def register_source_file(self, file_path: str, file_type: str = "audio", 
                           metadata: Dict[str, Any] = None) -> str:
        """Register a source file and return file_id"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        
        # Check if file already exists
        existing_id = self._get_file_id_by_hash(file_hash)
        if existing_id:
            return existing_id
        
        # Get file metadata
        file_size = file_path.stat().st_size
        file_format = file_path.suffix.lower().lstrip('.')
        
        # Create file ID
        file_id = f"{file_type}_{uuid.uuid4().hex[:8]}"
        
        # Copy file to storage
        stored_path = self.files_dir / f"{file_id}{file_path.suffix}"
        shutil.copy2(file_path, stored_path)
        
        # Create source file record
        source_file = SourceFile(
            file_id=file_id,
            original_path=str(file_path),
            file_type=file_type,
            file_size=file_size,
            duration=0.0,  # Will be updated after processing
            sample_rate=0,  # Will be updated after processing
            channels=0,     # Will be updated after processing
            format=file_format,
            hash=file_hash,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        # Save to database
        self._save_source_file(source_file)
        
        return file_id
    
    def update_source_file_metadata(self, file_id: str, duration: float, 
                                  sample_rate: int, channels: int, status: str = "completed"):
        """Update source file with processing results"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE source_files 
                SET duration = ?, sample_rate = ?, channels = ?, 
                    processed_at = ?, status = ?
                WHERE file_id = ?
            """, (duration, sample_rate, channels, datetime.now(), status, file_id))
            conn.commit()
    
    def save_audio_segments(self, file_id: str, segments: List[Dict[str, Any]]) -> List[str]:
        """Save audio segments and return segment IDs"""
        segment_ids = []
        
        for i, segment_data in enumerate(segments):
            segment_id = f"{file_id}_seg_{i:06d}"
            
            # Create audio segment record
            segment = AudioSegment(
                segment_id=segment_id,
                source_file_id=file_id,
                audio_file_path=segment_data['audio_file'],
                text=segment_data['text'],
                speaker_id=segment_data['speaker_id'],
                start_time=segment_data['start_time'],
                end_time=segment_data['end_time'],
                duration=segment_data['duration'],
                sample_rate=segment_data.get('sample_rate', 24000),
                quality_score=segment_data.get('quality_score', 0.0),
                confidence_score=segment_data.get('confidence_score', 0.0),
                language=segment_data.get('language', 'en'),
                created_at=datetime.now(),
                metadata=segment_data.get('metadata', {})
            )
            
            # Save to database
            self._save_audio_segment(segment)
            segment_ids.append(segment_id)
        
        return segment_ids
    
    def get_source_files(self, status: str = None) -> List[SourceFile]:
        """Get source files with optional status filter"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM source_files"
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status)
            
            query += " ORDER BY created_at DESC"
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            source_files = []
            for row in rows:
                source_files.append(SourceFile(
                    file_id=row[0],
                    original_path=row[1],
                    file_type=row[2],
                    file_size=row[3],
                    duration=row[4],
                    sample_rate=row[5],
                    channels=row[6],
                    format=row[7],
                    hash=row[8],
                    created_at=datetime.fromisoformat(row[9]),
                    processed_at=datetime.fromisoformat(row[10]) if row[10] else None,
                    status=row[11],
                    error_message=row[12],
                    metadata=json.loads(row[13]) if row[13] else {}
                ))
            
            return source_files
    
    def get_audio_segments(self, source_file_id: str = None, speaker_id: str = None) -> List[AudioSegment]:
        """Get audio segments with optional filters"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM audio_segments"
            params = []
            conditions = []
            
            if source_file_id:
                conditions.append("source_file_id = ?")
                params.append(source_file_id)
            
            if speaker_id:
                conditions.append("speaker_id = ?")
                params.append(speaker_id)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC"
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            segments = []
            for row in rows:
                segments.append(AudioSegment(
                    segment_id=row[0],
                    source_file_id=row[1],
                    audio_file_path=row[2],
                    text=row[3],
                    speaker_id=row[4],
                    start_time=row[5],
                    end_time=row[6],
                    duration=row[7],
                    sample_rate=row[8],
                    quality_score=row[9],
                    confidence_score=row[10],
                    language=row[11],
                    created_at=datetime.fromisoformat(row[12]),
                    metadata=json.loads(row[13]) if row[13] else {}
                ))
            
            return segments
    
    def create_dataset_export(self, dataset_name: str = None, 
                             output_format: str = "huggingface") -> str:
        """Create dataset export in specified format"""
        if dataset_name is None:
            dataset_name = f"tts_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        export_dir = self.base_dir / "exports" / dataset_name
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all segments
        segments = self.get_audio_segments()
        
        if output_format == "huggingface":
            return self._create_huggingface_export(export_dir, segments, dataset_name)
        elif output_format == "orpheus":
            return self._create_orpheus_export(export_dir, segments, dataset_name)
        else:
            raise ValueError(f"Unsupported export format: {output_format}")
    
    def _create_huggingface_export(self, export_dir: Path, segments: List[AudioSegment], 
                                  dataset_name: str) -> str:
        """Create Hugging Face dataset export"""
        # Create audio directory
        audio_dir = export_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # Copy audio files and create metadata
        metadata = []
        for i, segment in enumerate(segments):
            # Copy audio file
            new_audio_path = audio_dir / f"segment_{i:06d}.wav"
            if Path(segment.audio_file_path).exists():
                shutil.copy2(segment.audio_file_path, new_audio_path)
            
            # Create metadata entry
            metadata.append({
                "segment_id": segment.segment_id,
                "audio_file": str(new_audio_path),
                "text": segment.text,
                "speaker_id": segment.speaker_id,
                "duration": segment.duration,
                "sample_rate": segment.sample_rate,
                "language": segment.language,
                "quality_score": segment.quality_score,
                "confidence_score": segment.confidence_score,
                "source_file_id": segment.source_file_id,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "metadata": segment.metadata
            })
        
        # Save metadata
        metadata_file = export_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create dataset card
        self._create_dataset_card(export_dir, dataset_name, len(segments))
        
        return str(export_dir)
    
    def _create_orpheus_export(self, export_dir: Path, segments: List[AudioSegment], 
                              dataset_name: str) -> str:
        """Create Orpheus TTS dataset export"""
        # Create audio directory
        audio_dir = export_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # Create Orpheus metadata format
        orpheus_metadata = {
            "dataset_name": dataset_name,
            "total_segments": len(segments),
            "speakers": list(set(seg.speaker_id for seg in segments)),
            "segments": []
        }
        
        for i, segment in enumerate(segments):
            # Copy audio file
            new_audio_path = audio_dir / f"segment_{i:06d}.wav"
            if Path(segment.audio_file_path).exists():
                shutil.copy2(segment.audio_file_path, new_audio_path)
            
            # Add to Orpheus format
            orpheus_metadata["segments"].append({
                "audio_file": str(new_audio_path),
                "text": segment.text,
                "speaker": segment.speaker_id,
                "duration": segment.duration
            })
        
        # Save Orpheus metadata
        metadata_file = export_dir / "orpheus_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(orpheus_metadata, f, indent=2)
        
        return str(export_dir)
    
    def _create_dataset_card(self, export_dir: Path, dataset_name: str, total_segments: int):
        """Create dataset card for Hugging Face"""
        card_content = f"""---
language:
- en
pretty_name: "{dataset_name}"
tags:
- audio
- speech
- tts
- transcription
- english
license: "mit"
task_categories:
- text-to-speech
- automatic-speech-recognition
size_categories:
- 1K<n<10K
---

# {dataset_name}

This dataset contains {total_segments} audio segments with transcriptions for Text-to-Speech training.

## Dataset Structure
- `audio/`: Audio files (WAV format, 24kHz)
- `metadata.json`: Contains segment metadata

## Usage
```python
import json

# Load metadata
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

# Access segments
for segment in metadata:
    print(f"Text: {{segment['text']}}")
    print(f"Speaker: {{segment['speaker_id']}}")
    print(f"Audio: {{segment['audio_file']}}")
```

## Statistics
- Total segments: {total_segments}
- Language: English
- Sample rate: 24kHz
- Format: WAV
"""
        
        readme_file = export_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(card_content)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_file_id_by_hash(self, file_hash: str) -> Optional[str]:
        """Get file ID by hash if file already exists"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT file_id FROM source_files WHERE hash = ?", (file_hash,))
            row = cursor.fetchone()
            return row[0] if row else None
    
    def _save_source_file(self, source_file: SourceFile):
        """Save source file to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO source_files 
                (file_id, original_path, file_type, file_size, duration, sample_rate, 
                 channels, format, hash, created_at, processed_at, status, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                source_file.file_id,
                source_file.original_path,
                source_file.file_type,
                source_file.file_size,
                source_file.duration,
                source_file.sample_rate,
                source_file.channels,
                source_file.format,
                source_file.hash,
                source_file.created_at.isoformat(),
                source_file.processed_at.isoformat() if source_file.processed_at else None,
                source_file.status,
                source_file.error_message,
                json.dumps(source_file.metadata) if source_file.metadata else None
            ))
            conn.commit()
    
    def _save_audio_segment(self, segment: AudioSegment):
        """Save audio segment to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO audio_segments 
                (segment_id, source_file_id, audio_file_path, text, speaker_id, 
                 start_time, end_time, duration, sample_rate, quality_score, 
                 confidence_score, language, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                segment.segment_id,
                segment.source_file_id,
                segment.audio_file_path,
                segment.text,
                segment.speaker_id,
                segment.start_time,
                segment.end_time,
                segment.duration,
                segment.sample_rate,
                segment.quality_score,
                segment.confidence_score,
                segment.language,
                segment.created_at.isoformat(),
                json.dumps(segment.metadata) if segment.metadata else None
            ))
            conn.commit()
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Get source file stats
            cursor = conn.execute("""
                SELECT COUNT(*) as total_files,
                       COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_files,
                       COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_files,
                       SUM(file_size) as total_size
                FROM source_files
            """)
            file_stats = cursor.fetchone()
            
            # Get segment stats
            cursor = conn.execute("""
                SELECT COUNT(*) as total_segments,
                       COUNT(DISTINCT speaker_id) as unique_speakers,
                       SUM(duration) as total_duration,
                       AVG(quality_score) as avg_quality,
                       AVG(confidence_score) as avg_confidence
                FROM audio_segments
            """)
            segment_stats = cursor.fetchone()
            
            # Get speaker distribution
            cursor = conn.execute("""
                SELECT speaker_id, COUNT(*) as segment_count, SUM(duration) as total_duration
                FROM audio_segments
                GROUP BY speaker_id
                ORDER BY segment_count DESC
            """)
            speaker_dist = cursor.fetchall()
            
            return {
                "source_files": {
                    "total": file_stats[0],
                    "completed": file_stats[1],
                    "failed": file_stats[2],
                    "total_size_bytes": file_stats[3] or 0
                },
                "audio_segments": {
                    "total": segment_stats[0],
                    "unique_speakers": segment_stats[1],
                    "total_duration_seconds": segment_stats[2] or 0,
                    "average_quality": segment_stats[3] or 0,
                    "average_confidence": segment_stats[4] or 0
                },
                "speaker_distribution": [
                    {"speaker_id": row[0], "segment_count": row[1], "total_duration": row[2]}
                    for row in speaker_dist
                ]
            }
