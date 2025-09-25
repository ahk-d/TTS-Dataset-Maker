"""
Simple configuration management using Pydantic Settings
"""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # AssemblyAI Configuration
    assemblyai_api_key: Optional[str] = Field(None, env="ASSEMBLYAI_API_KEY")
    
    # Label Studio Configuration
    label_studio_url: str = Field("http://localhost:8080", env="LABEL_STUDIO_URL")
    label_studio_api_key: Optional[str] = Field(None, env="LABEL_STUDIO_API_KEY")
    
    # Denoising Configuration
    denoising_enabled: bool = Field(True, env="DENOISING_ENABLED")
    denoising_chunk_duration: float = Field(30.0, env="DENOISING_CHUNK_DURATION")
    denoising_overlap_duration: float = Field(2.0, env="DENOISING_OVERLAP_DURATION")
    denoising_sample_rate: int = Field(48000, env="DENOISING_SAMPLE_RATE")
    
    # Voice Activity Detection Configuration
    vad_enabled: bool = Field(True, env="VAD_ENABLED")
    vad_method: str = Field("silero", env="VAD_METHOD")
    min_speech_duration: float = Field(0.5, env="MIN_SPEECH_DURATION")
    silence_threshold: float = Field(-30.0, env="SILENCE_THRESHOLD")
    min_speech_ratio: float = Field(0.3, env="MIN_SPEECH_RATIO")
    
    # Silence Removal Configuration
    remove_long_silences: bool = Field(True, env="REMOVE_LONG_SILENCES")
    max_silence_duration: float = Field(1.0, env="MAX_SILENCE_DURATION")
    silence_padding_duration: float = Field(0.1, env="SILENCE_PADDING_DURATION")
    
    # Quality Settings
    min_segment_duration: float = Field(1.0, env="MIN_SEGMENT_DURATION")
    max_segment_duration: float = Field(3600.0, env="MAX_SEGMENT_DURATION")
    min_confidence_score: float = Field(0.7, env="MIN_CONFIDENCE_SCORE")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Clean API key by removing quotes
        if self.assemblyai_api_key:
            self.assemblyai_api_key = self.assemblyai_api_key.strip('"\'')
    
    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()