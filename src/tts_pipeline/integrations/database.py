"""
Modern database integration using SQLAlchemy 2.0+
Converted from existing production_storage.py
"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.sqlite import insert
import logging

from ..config.settings import settings
from ..models.audio import AudioFile, AudioSegment
from ..models.dataset import DatasetMetadata, DatasetStatistics


Base = declarative_base()


class SourceFileModel(Base):
    """SQLAlchemy model for source files"""
    __tablename__ = "source_files"
    
    file_id = Column(String, primary_key=True)
    original_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer)
    duration = Column(Float)
    sample_rate = Column(Integer)
    channels = Column(Integer)
    format = Column(String)
    hash = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    status = Column(String, default="pending")
    error_message = Column(Text)
    metadata_json = Column(Text)  # JSON string


class AudioSegmentModel(Base):
    """SQLAlchemy model for audio segments"""
    __tablename__ = "audio_segments"
    
    segment_id = Column(String, primary_key=True)
    source_file_id = Column(String, nullable=False)
    audio_file_path = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    speaker_id = Column(String, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    sample_rate = Column(Integer)
    quality_score = Column(Float)
    confidence_score = Column(Float)
    language = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(Text)  # JSON string


class DatabaseManager:
    """Modern database manager with SQLAlchemy 2.0+"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or "output/datasets")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create database URL
        db_path = self.base_dir / "database.db"
        self.database_url = f"sqlite:///{db_path}"
        
        # Create engine and session
        self.engine = create_engine(self.database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Database initialized: {self.database_url}")
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    async def register_source_file(
        self, 
        file_path: str, 
        file_type: str = "audio",
        metadata: Dict[str, Any] = None
    ) -> str:
        """Register a source file and return file_id"""
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        
        # Check if file already exists
        with self.get_session() as session:
            existing = session.query(SourceFileModel).filter_by(hash=file_hash).first()
            if existing:
                return existing.file_id
        
        # Create file ID
        file_id = f"{file_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_path.stem}"
        
        # Get file metadata
        file_size = file_path.stat().st_size
        file_format = file_path.suffix.lower().lstrip('.')
        
        # Create source file record
        source_file = SourceFileModel(
            file_id=file_id,
            original_path=str(file_path),
            file_type=file_type,
            file_size=file_size,
            duration=0.0,  # Will be updated after processing
            sample_rate=0,  # Will be updated after processing
            channels=0,     # Will be updated after processing
            format=file_format,
            hash=file_hash,
            metadata_json=str(metadata) if metadata else None
        )
        
        # Save to database
        with self.get_session() as session:
            session.add(source_file)
            session.commit()
        
        self.logger.info(f"Registered source file: {file_id}")
        return file_id
    
    async def update_source_file_metadata(
        self, 
        file_id: str, 
        duration: float,
        sample_rate: int, 
        channels: int, 
        status: str = "completed"
    ):
        """Update source file with processing results"""
        
        with self.get_session() as session:
            source_file = session.query(SourceFileModel).filter_by(file_id=file_id).first()
            if source_file:
                source_file.duration = duration
                source_file.sample_rate = sample_rate
                source_file.channels = channels
                source_file.status = status
                source_file.processed_at = datetime.utcnow()
                session.commit()
    
    async def save_audio_segments(
        self, 
        file_id: str, 
        segments: List[Dict[str, Any]]
    ) -> List[str]:
        """Save audio segments and return segment IDs"""
        
        segment_ids = []
        
        with self.get_session() as session:
            for i, segment_data in enumerate(segments):
                segment_id = f"{file_id}_seg_{i:06d}"
                
                segment = AudioSegmentModel(
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
                    metadata_json=str(segment_data.get('metadata', {}))
                )
                
                session.add(segment)
                segment_ids.append(segment_id)
            
            session.commit()
        
        self.logger.info(f"Saved {len(segment_ids)} audio segments")
        return segment_ids
    
    def get_source_files(self, status: str = None) -> List[AudioFile]:
        """Get source files with optional status filter"""
        
        with self.get_session() as session:
            query = session.query(SourceFileModel)
            
            if status:
                query = query.filter_by(status=status)
            
            source_files = query.order_by(SourceFileModel.created_at.desc()).all()
            
            return [
                AudioFile(
                    file_id=sf.file_id,
                    original_path=sf.original_path,
                    file_type=sf.file_type,
                    file_size=sf.file_size,
                    duration=sf.duration,
                    sample_rate=sf.sample_rate,
                    channels=sf.channels,
                    format=sf.format,
                    hash=sf.hash,
                    created_at=sf.created_at,
                    processed_at=sf.processed_at,
                    status=sf.status,
                    error_message=sf.error_message,
                    metadata=eval(sf.metadata_json) if sf.metadata_json else {}
                )
                for sf in source_files
            ]
    
    def get_audio_segments(
        self, 
        source_file_id: str = None, 
        speaker_id: str = None
    ) -> List[AudioSegment]:
        """Get audio segments with optional filters"""
        
        with self.get_session() as session:
            query = session.query(AudioSegmentModel)
            
            if source_file_id:
                query = query.filter_by(source_file_id=source_file_id)
            
            if speaker_id:
                query = query.filter_by(speaker_id=speaker_id)
            
            segments = query.order_by(AudioSegmentModel.created_at.desc()).all()
            
            return [
                AudioSegment(
                    segment_id=s.segment_id,
                    source_file_id=s.source_file_id,
                    audio_file_path=s.audio_file_path,
                    text=s.text,
                    speaker_id=s.speaker_id,
                    start_time=s.start_time,
                    end_time=s.end_time,
                    duration=s.duration,
                    sample_rate=s.sample_rate,
                    quality_score=s.quality_score,
                    confidence_score=s.confidence_score,
                    language=s.language,
                    created_at=s.created_at,
                    metadata=eval(s.metadata_json) if s.metadata_json else {}
                )
                for s in segments
            ]
    
    def get_dataset_statistics(self) -> DatasetStatistics:
        """Get comprehensive dataset statistics"""
        
        with self.get_session() as session:
            # Source file stats
            total_files = session.query(SourceFileModel).count()
            completed_files = session.query(SourceFileModel).filter_by(status="completed").count()
            failed_files = session.query(SourceFileModel).filter_by(status="failed").count()
            
            # Segment stats
            total_segments = session.query(AudioSegmentModel).count()
            unique_speakers = session.query(AudioSegmentModel.speaker_id).distinct().count()
            
            # Calculate averages
            avg_quality = session.query(AudioSegmentModel.quality_score).filter(
                AudioSegmentModel.quality_score.isnot(None)
            ).all()
            avg_quality = sum(q[0] for q in avg_quality) / len(avg_quality) if avg_quality else 0.0
            
            avg_confidence = session.query(AudioSegmentModel.confidence_score).filter(
                AudioSegmentModel.confidence_score.isnot(None)
            ).all()
            avg_confidence = sum(c[0] for c in avg_confidence) / len(avg_confidence) if avg_confidence else 0.0
            
            # Total duration
            total_duration = session.query(AudioSegmentModel.duration).all()
            total_duration = sum(d[0] for d in total_duration) if total_duration else 0.0
            
            return DatasetStatistics(
                total_files=total_files,
                processed_files=completed_files,
                failed_files=failed_files,
                total_segments=total_segments,
                total_duration=total_duration,
                unique_speakers=unique_speakers,
                average_quality=avg_quality,
                average_confidence=avg_confidence,
                processing_time=0.0,  # TODO: Track processing time
                success_rate=completed_files / total_files if total_files > 0 else 0.0
            )
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        import hashlib
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
