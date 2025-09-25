"""
AssemblyAI integration for transcription and speaker diarization
Using the official assemblyai package
"""
import asyncio
import logging
from typing import Dict, List, Optional
import assemblyai as aai

from ..config.settings import settings

logger = logging.getLogger(__name__)


class DiarizationResult:
    """Result from speaker diarization"""
    def __init__(self, utterances: List[Dict], speakers: List[str], confidence: float):
        self.utterances = utterances
        self.speakers = speakers
        self.confidence = confidence


class AssemblyAIClient:
    """Client for AssemblyAI transcription and speaker diarization"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()
        logger.info("AssemblyAI client initialized")
    
    async def submit_transcription(self, audio_file: str, language: str = "en") -> str:
        """Submit audio file for transcription"""
        
        # Configure transcription settings
        config = aai.TranscriptionConfig(
            speaker_labels=True,  # Enable speaker diarization
            language_code=language,
            punctuate=True,
            format_text=True,
            dual_channel=False,
            boost_param="high"
        )
        
        logger.info(f"Starting transcription for {audio_file}")
        
        # Transcribe with AssemblyAI
        transcript = self.transcriber.transcribe(audio_file, config=config)
        
        # Wait for completion
        while transcript.status not in [aai.TranscriptStatus.completed, aai.TranscriptStatus.error]:
            await asyncio.sleep(1)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")
        
        logger.info(f"Transcription completed successfully")
        return transcript
    
    async def wait_for_completion(self, transcript) -> Dict:
        """Wait for transcription to complete"""
        return transcript
    
    async def get_transcript(self, transcript) -> Dict:
        """Get transcription result"""
        return transcript
    
    def transcribe_audio(self, audio_file: str, language: str = "en") -> Dict:
        """Synchronous transcription method for batch processing"""
        import asyncio
        
        # Run the async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            transcript = loop.run_until_complete(self.submit_transcription(audio_file, language))
            return transcript
        finally:
            loop.close()
    
    def extract_speaker_segments(self, transcript) -> List[Dict]:
        """Extract speaker segments from transcript"""
        segments = []
        
        if not transcript or not hasattr(transcript, 'utterances'):
            logger.warning("No utterances found in transcript")
            return segments
        
        for utterance in transcript.utterances:
            if utterance.speaker:
                segment = {
                    'text': utterance.text,
                    'speaker': utterance.speaker,
                    'start_time': utterance.start / 1000.0,  # Convert ms to seconds
                    'end_time': utterance.end / 1000.0,      # Convert ms to seconds
                    'confidence': utterance.confidence,
                    'audio_file': None  # Will be set during audio processing
                }
                segments.append(segment)
        
        logger.info(f"Extracted {len(segments)} speaker segments")
        return segments