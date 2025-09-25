"""
Configuration settings for TTS Dataset Maker
"""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # AssemblyAI Configuration
    assemblyai_api_key: Optional[str] = Field(None, env="ASSEMBLYAI_API_KEY")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Clean API key by removing quotes
        if self.assemblyai_api_key:
            self.assemblyai_api_key = self.assemblyai_api_key.strip('"\'')
    
    # Audio Processing Configuration
    default_sample_rate: int = Field(24000, env="DEFAULT_SAMPLE_RATE")
    min_segment_duration: float = Field(1.0, env="MIN_SEGMENT_DURATION")
    max_segment_duration: float = Field(3600.0, env="MAX_SEGMENT_DURATION")  # 1 hour
    min_confidence_score: float = Field(0.7, env="MIN_CONFIDENCE_SCORE")
    audio_overlap_ms: int = Field(100, env="AUDIO_OVERLAP_MS")
    
    # Output Configuration
    output_dir: str = Field("./output", env="OUTPUT_DIR")
    temp_dir: str = Field("./temp", env="TEMP_DIR")
    
    # Processing Configuration
    max_speakers: Optional[int] = Field(None, env="MAX_SPEAKERS")
    enable_auto_highlights: bool = Field(True, env="ENABLE_AUTO_HIGHLIGHTS")
    enable_sentiment_analysis: bool = Field(False, env="ENABLE_SENTIMENT_ANALYSIS")
    enable_entity_detection: bool = Field(False, env="ENABLE_ENTITY_DETECTION")
    
    # Multi-file Processing Configuration
    batch_size: int = Field(1, env="BATCH_SIZE")  # Process files one at a time by default
    parallel_processing: bool = Field(False, env="PARALLEL_PROCESSING")  # Future: parallel processing
    max_concurrent_files: int = Field(1, env="MAX_CONCURRENT_FILES")  # Future: concurrent processing
    preserve_temp_files: bool = Field(False, env="PRESERVE_TEMP_FILES")  # Keep temp files for debugging
    speaker_mapping_mode: str = Field("interactive", env="SPEAKER_MAPPING_MODE")  # interactive, auto, manual
    
    # Denoising removed - no longer needed
    
    # Voice Activity Detection Configuration
    vad_enabled: bool = Field(True, env="VAD_ENABLED")
    vad_method: str = Field("webrtcvad", env="VAD_METHOD")
    min_speech_duration: float = Field(0.5, env="MIN_SPEECH_DURATION")
    silence_threshold: float = Field(-40.0, env="SILENCE_THRESHOLD")
    min_speech_ratio: float = Field(0.3, env="MIN_SPEECH_RATIO")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
