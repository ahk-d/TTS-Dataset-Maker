"""
Dataset-related Pydantic models
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DatasetMetadata(BaseModel):
    """Dataset metadata"""
    dataset_name: str
    total_segments: int
    total_duration: float
    speakers: List[str]
    languages: List[str]
    created_at: datetime
    updated_at: datetime
    version: str = "1.0"
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetStatistics(BaseModel):
    """Dataset statistics"""
    total_files: int
    processed_files: int
    failed_files: int
    total_segments: int
    total_duration: float
    unique_speakers: int
    average_quality: float
    average_confidence: float
    processing_time: float
    success_rate: float


class DatasetExport(BaseModel):
    """Dataset export configuration"""
    format: str  # huggingface, orpheus, custom
    output_path: str
    include_metadata: bool = True
    include_quality_scores: bool = True
    include_confidence_scores: bool = True
    compression: Optional[str] = None
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessingSession(BaseModel):
    """Processing session tracking"""
    session_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_files: int
    processed_files: int
    failed_files: int
    total_segments: int
    status: str  # running, completed, failed
    metadata: Dict[str, Any] = Field(default_factory=dict)
