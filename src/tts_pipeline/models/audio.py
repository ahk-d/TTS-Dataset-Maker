"""
Audio-related Pydantic models
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class AudioFile(BaseModel):
    """Audio file metadata"""
    file_id: str
    original_path: str
    file_type: str = "audio"
    file_size: int
    duration: float
    sample_rate: int
    channels: int
    format: str
    hash: str
    created_at: datetime
    processed_at: Optional[datetime] = None
    status: str = "pending"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AudioSegment(BaseModel):
    """Audio segment with full metadata"""
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
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AudioQualityMetrics(BaseModel):
    """Audio quality assessment metrics"""
    signal_to_noise_ratio: float
    spectral_centroid: float
    zero_crossing_rate: float
    mfcc_features: list[float]
    rms_energy: float
    spectral_rolloff: float
    quality_score: float
    confidence_score: float


class VoiceActivityDetection(BaseModel):
    """Voice activity detection results"""
    speech_segments: list[tuple[float, float]]  # (start, end) tuples
    total_speech_duration: float
    total_silence_duration: float
    speech_ratio: float
    is_speech_present: bool
