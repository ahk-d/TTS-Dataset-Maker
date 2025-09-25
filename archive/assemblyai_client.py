"""
AssemblyAI integration for transcription and speaker diarization
Updated for latest AssemblyAI API with automatic speaker detection
"""
import os
import logging
import time
import hashlib
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import assemblyai as aai
from pydantic import BaseModel

from config import settings


logger = logging.getLogger(__name__)


class SpeakerSegment(BaseModel):
    """Model for speaker segment data"""
    speaker: str
    text: str
    start: float
    end: float
    confidence: float
    words: Optional[List[Dict]] = None


class TranscriptionResult(BaseModel):
    """Model for transcription result"""
    segments: List[SpeakerSegment]
    full_text: str
    confidence: float
    language_code: Optional[str] = None
    processing_time: float


class AssemblyAIClient:
    """Client for AssemblyAI transcription and speaker diarization"""
    
    def __init__(self, api_key: str = None, cache_dir: str = None):
        """
        Initialize AssemblyAI client
        
        Args:
            api_key: AssemblyAI API key (defaults to settings)
            cache_dir: Directory to store transcription cache
        """
        self.api_key = api_key or settings.assemblyai_api_key
        aai.settings.api_key = self.api_key
        
        # Setup cache directory
        self.cache_dir = cache_dir or os.path.join(settings.output_dir, "transcription_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Configure transcription settings for optimal automatic speaker diarization
        self.config = aai.TranscriptionConfig(
            # Core speaker diarization - automatic detection
            speaker_labels=True,  # Enable automatic speaker diarization
            
            # Language settings
            language_code="en",  # Default to English
            language_detection=True,  # Auto-detect language if needed
            
            # Text formatting
            punctuate=True,
            format_text=True,
            
            # Audio processing
            dual_channel=False,  # We convert to mono
            
            # Additional features (optional based on settings)
            auto_highlights=getattr(settings, 'enable_auto_highlights', False),
            sentiment_analysis=getattr(settings, 'enable_sentiment_analysis', False),
            entity_detection=getattr(settings, 'enable_entity_detection', False),
            
            # Quality settings
            boost_param="high",  # Boost accuracy for better speaker detection
            filter_profanity=False,  # Keep all content for TTS training
            
            # No webhook for synchronous processing
            webhook_url=None,
        )
        
        self.transcriber = aai.Transcriber()
        logger.info(f"AssemblyAI client initialized with cache dir: {self.cache_dir}")
    
    def _get_cache_key(self, file_path: str, language: str = "en") -> str:
        """
        Generate cache key for file
        
        Args:
            file_path: Path to audio file
            language: Language code
            
        Returns:
            Cache key string
        """
        # Get file stats for cache invalidation
        stat = os.stat(file_path)
        file_info = f"{file_path}_{stat.st_size}_{stat.st_mtime}_{language}"
        return hashlib.md5(file_info.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get cache file path
        
        Args:
            cache_key: Cache key
            
        Returns:
            Path to cache file
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _load_from_cache(self, cache_key: str) -> Optional[TranscriptionResult]:
        """
        Load transcription from cache
        
        Args:
            cache_key: Cache key
            
        Returns:
            TranscriptionResult if found, None otherwise
        """
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Convert back to TranscriptionResult
            segments = [SpeakerSegment(**seg) for seg in cache_data['segments']]
            result = TranscriptionResult(
                segments=segments,
                full_text=cache_data['full_text'],
                confidence=cache_data['confidence'],
                language_code=cache_data.get('language_code'),
                processing_time=cache_data.get('processing_time', 0.0)
            )
            
            logger.info(f"Loaded transcription from cache: {cache_path}")
            return result
            
        except Exception as e:
            logger.warning(f"Failed to load from cache {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, result: TranscriptionResult) -> None:
        """
        Save transcription to cache
        
        Args:
            cache_key: Cache key
            result: TranscriptionResult to cache
        """
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Convert to JSON-serializable format
            cache_data = {
                'segments': [seg.dict() for seg in result.segments],
                'full_text': result.full_text,
                'confidence': result.confidence,
                'language_code': result.language_code,
                'processing_time': result.processing_time,
                'cached_at': time.time()
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved transcription to cache: {cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save to cache {cache_path}: {e}")
    
    def upload_audio(self, file_path: str) -> str:
        """
        Upload audio file to AssemblyAI
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Upload URL
        """
        try:
            logger.info(f"Uploading audio file: {file_path}")
            upload_url = aai.upload_file(file_path)  # Updated API call
            logger.info(f"Audio uploaded successfully: {upload_url}")
            return upload_url
        except Exception as e:
            logger.error(f"Failed to upload audio file {file_path}: {e}")
            raise
    
    def configure_language(self, language: str = None, auto_detect: bool = True):
        """
        Configure language settings for transcription
        
        Args:
            language: Specific language code (e.g., 'en', 'es', 'fr')
            auto_detect: Whether to enable automatic language detection
        """
        if auto_detect:
            self.config.language_detection = True
            self.config.language_code = None  # Let API auto-detect
            logger.info("Enabled automatic language detection")
        else:
            self.config.language_detection = False
            self.config.language_code = language or "en"
            logger.info(f"Set language to: {self.config.language_code}")
    
    def transcribe_audio(self, audio_source: str, language: str = None) -> aai.Transcript:
        """
        Transcribe audio with automatic speaker diarization
        
        Args:
            audio_source: Path to audio file or URL
            language: Language code (optional, will auto-detect if None)
            
        Returns:
            AssemblyAI Transcript object
        """
        try:
            # Update language configuration if specified
            if language:
                self.configure_language(language, auto_detect=False)
            else:
                self.configure_language(auto_detect=True)
            
            logger.info(f"Starting transcription with automatic speaker diarization for: {audio_source}")
            start_time = time.time()
            
            # Transcribe with automatic speaker diarization
            transcript = self.transcriber.transcribe(
                audio_source,
                config=self.config
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f}s")
            
            # Log speaker information
            if hasattr(transcript, 'utterances') and transcript.utterances:
                speakers = set(utterance.speaker for utterance in transcript.utterances)
                logger.info(f"Automatically detected {len(speakers)} speakers: {', '.join(sorted(speakers))}")
            
            return transcript
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_source}: {e}")
            raise
    
    def extract_speaker_segments(self, transcript: aai.Transcript) -> List[SpeakerSegment]:
        """
        Extract speaker segments from transcript
        
        Args:
            transcript: AssemblyAI Transcript object
            
        Returns:
            List of SpeakerSegment objects
        """
        segments = []
        
        try:
            if not hasattr(transcript, 'utterances') or not transcript.utterances:
                logger.warning("No utterances found in transcript - speaker diarization may not be available")
                # Fallback: create single segment from full text if available
                if transcript.text:
                    segment = SpeakerSegment(
                        speaker="Unknown",
                        text=transcript.text.strip(),
                        start=0.0,
                        end=getattr(transcript, 'audio_duration', 0.0) / 1000.0,
                        confidence=transcript.confidence or 0.0,
                        words=None
                    )
                    segments.append(segment)
                return segments
            
            for utterance in transcript.utterances:
                # Extract word-level timestamps if available
                words = []
                if hasattr(utterance, 'words') and utterance.words:
                    words = [
                        {
                            "text": word.text,
                            "start": word.start / 1000.0,  # Convert to seconds
                            "end": word.end / 1000.0,      # Convert to seconds
                            "confidence": word.confidence
                        }
                        for word in utterance.words
                    ]
                
                segment = SpeakerSegment(
                    speaker=utterance.speaker,
                    text=utterance.text.strip(),
                    start=utterance.start / 1000.0,  # Convert to seconds
                    end=utterance.end / 1000.0,      # Convert to seconds
                    confidence=utterance.confidence,
                    words=words if words else None
                )
                
                segments.append(segment)
            
            logger.info(f"Extracted {len(segments)} speaker segments")
            return segments
            
        except Exception as e:
            logger.error(f"Failed to extract speaker segments: {e}")
            raise
    
    def process_audio_file(self, file_path: str, language: str = None) -> TranscriptionResult:
        """
        Complete processing pipeline for audio file with caching
        
        Args:
            file_path: Path to audio file
            language: Language code (optional, will auto-detect if None)
            
        Returns:
            TranscriptionResult object
        """
        try:
            logger.info(f"Processing audio file: {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Check cache first
            cache_key = self._get_cache_key(file_path, language or "auto")
            cached_result = self._load_from_cache(cache_key)
            
            if cached_result:
                logger.info(f"Using cached transcription for {file_path}")
                return cached_result
            
            # Transcribe audio
            start_time = time.time()
            transcript = self.transcribe_audio(file_path, language)
            processing_time = time.time() - start_time
            
            # Check if transcription was successful
            if transcript.status != aai.TranscriptStatus.completed:
                error_msg = f"Transcription failed with status: {transcript.status}"
                if hasattr(transcript, 'error'):
                    error_msg += f" - Error: {transcript.error}"
                raise Exception(error_msg)
            
            # Extract speaker segments
            segments = self.extract_speaker_segments(transcript)
            
            # Determine the actual language used
            detected_language = getattr(transcript, 'language_code', language or 'en')
            
            # Create result
            result = TranscriptionResult(
                segments=segments,
                full_text=transcript.text or "",
                confidence=transcript.confidence or 0.0,
                language_code=detected_language,
                processing_time=processing_time
            )
            
            # Save to cache
            self._save_to_cache(cache_key, result)
            
            # Log summary
            speakers = set(seg.speaker for seg in segments)
            logger.info(f"Successfully processed {file_path}: {len(segments)} segments from {len(speakers)} speakers")
            logger.info(f"Detected language: {detected_language}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process audio file {file_path}: {e}")
            raise
    
    def validate_transcription_quality(self, result: TranscriptionResult) -> Tuple[bool, List[str]]:
        """
        Validate transcription quality
        
        Args:
            result: TranscriptionResult object
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check overall confidence
        min_confidence = getattr(settings, 'min_confidence_score', 0.5)
        if result.confidence < min_confidence:
            issues.append(f"Low overall confidence: {result.confidence:.2f} < {min_confidence}")
        
        # Check segment quality
        low_confidence_segments = 0
        empty_segments = 0
        short_segments = 0
        
        min_segment_duration = getattr(settings, 'min_segment_duration', 1.0)
        
        for segment in result.segments:
            # Check confidence
            if segment.confidence < min_confidence:
                low_confidence_segments += 1
            
            # Check text content
            if not segment.text.strip():
                empty_segments += 1
            
            # Check duration
            duration = segment.end - segment.start
            if duration < min_segment_duration:
                short_segments += 1
        
        if low_confidence_segments > 0:
            issues.append(f"{low_confidence_segments} segments with low confidence")
        
        if empty_segments > 0:
            issues.append(f"{empty_segments} empty segments")
        
        if short_segments > 0:
            issues.append(f"{short_segments} segments too short (< {min_segment_duration}s)")
        
        # Check speaker diversity
        speakers = set(segment.speaker for segment in result.segments)
        if len(speakers) < 1:
            issues.append("No speakers detected")
        elif len(speakers) > 20:  # More reasonable upper limit
            issues.append(f"Unusually high number of speakers detected: {len(speakers)} (may indicate poor diarization)")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_speaker_statistics(self, result: TranscriptionResult) -> Dict[str, Dict]:
        """
        Get statistics for each speaker
        
        Args:
            result: TranscriptionResult object
            
        Returns:
            Dictionary with speaker statistics
        """
        speaker_stats = {}
        
        for segment in result.segments:
            speaker = segment.speaker
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "total_segments": 0,
                    "total_duration": 0.0,
                    "total_text_length": 0,
                    "total_words": 0,
                    "avg_confidence": 0.0,
                    "segments": []
                }
            
            stats = speaker_stats[speaker]
            duration = segment.end - segment.start
            word_count = len(segment.text.split())
            
            stats["total_segments"] += 1
            stats["total_duration"] += duration
            stats["total_text_length"] += len(segment.text)
            stats["total_words"] += word_count
            stats["segments"].append({
                "start": segment.start,
                "end": segment.end,
                "duration": duration,
                "confidence": segment.confidence,
                "text_length": len(segment.text),
                "word_count": word_count
            })
        
        # Calculate averages and percentages
        total_duration = sum(stats["total_duration"] for stats in speaker_stats.values())
        
        for speaker, stats in speaker_stats.items():
            if stats["total_segments"] > 0:
                confidences = [s["confidence"] for s in stats["segments"]]
                stats["avg_confidence"] = sum(confidences) / len(confidences)
                stats["avg_segment_duration"] = stats["total_duration"] / stats["total_segments"]
                stats["avg_text_length"] = stats["total_text_length"] / stats["total_segments"]
                stats["avg_words_per_segment"] = stats["total_words"] / stats["total_segments"]
                
                # Calculate speaking time percentage
                if total_duration > 0:
                    stats["speaking_time_percentage"] = (stats["total_duration"] / total_duration) * 100
                else:
                    stats["speaking_time_percentage"] = 0.0
        
        return speaker_stats
    
    def get_transcription_summary(self, result: TranscriptionResult) -> Dict:
        """
        Get a summary of the transcription results
        
        Args:
            result: TranscriptionResult object
            
        Returns:
            Dictionary with summary statistics
        """
        speaker_stats = self.get_speaker_statistics(result)
        
        total_duration = sum(seg.end - seg.start for seg in result.segments)
        total_words = sum(len(seg.text.split()) for seg in result.segments)
        
        summary = {
            "total_segments": len(result.segments),
            "total_speakers": len(speaker_stats),
            "total_duration_seconds": total_duration,
            "total_words": total_words,
            "average_confidence": result.confidence,
            "language": result.language_code,
            "processing_time_seconds": result.processing_time,
            "speakers": list(speaker_stats.keys()),
            "words_per_minute": (total_words / (total_duration / 60)) if total_duration > 0 else 0,
        }
        
        return summary