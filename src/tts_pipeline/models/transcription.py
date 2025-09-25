"""
Transcription-related Pydantic models
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class Speaker(BaseModel):
    """Speaker information"""
    speaker_id: str
    name: Optional[str] = None
    confidence: float
    total_duration: float
    segment_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TranscriptionWord(BaseModel):
    """Individual word in transcription"""
    text: str
    start: float
    end: float
    confidence: float
    speaker: Optional[str] = None


class TranscriptionUtterance(BaseModel):
    """Utterance with speaker information"""
    text: str
    start: float
    end: float
    confidence: float
    speaker: str
    words: List[TranscriptionWord] = Field(default_factory=list)


class TranscriptionResult(BaseModel):
    """Complete transcription result"""
    text: str
    confidence: float
    language: str
    utterances: List[TranscriptionUtterance]
    speakers: List[Speaker]
    total_duration: float
    processing_time: float
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DiarizationResult(BaseModel):
    """Speaker diarization results"""
    speakers: List[Speaker]
    utterances: List[TranscriptionUtterance]
    speaker_turns: List[tuple[str, float, float]]  # (speaker, start, end)
    speaker_changes: int
    average_speaker_duration: float
    confidence: float
