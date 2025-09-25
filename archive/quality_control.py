"""
Quality control module for TTS Dataset Maker
Implements filtering, validation, and quality assessment mechanisms
"""
import os
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import re

from pydantic import BaseModel

from audio_segmenter import AudioSegmentData, SegmentationResult
from config import settings


logger = logging.getLogger(__name__)


class QualityReport(BaseModel):
    """Quality control report"""
    total_segments: int
    passed_segments: int
    failed_segments: int
    quality_issues: Dict[str, List[str]]
    statistics: Dict[str, Any]
    recommendations: List[str]


class QualityFilter:
    """Quality control and filtering for TTS training data"""
    
    def __init__(self):
        """Initialize quality filter"""
        self.min_confidence = settings.min_confidence_score
        self.min_duration = settings.min_segment_duration
        self.max_duration = settings.max_segment_duration
        self.min_text_length = 10
        self.max_text_length = 500
        
        # Common filler words and noise patterns
        self.filler_words = {
            'um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'well', 'actually',
            'basically', 'literally', 'right', 'okay', 'ok', 'yeah', 'yes', 'no'
        }
        
        # Noise patterns in text
        self.noise_patterns = [
            r'\[.*?\]',  # Bracketed content
            r'\(.*?\)',  # Parenthesized content
            r'<.*?>',    # Angle brackets
            r'[^\w\s.,!?;:\'-]',  # Special characters
            r'\b\d+\b',  # Standalone numbers
        ]
        
        logger.info("Quality filter initialized")
    
    def validate_audio_quality(self, segment: AudioSegmentData) -> Tuple[bool, List[str]]:
        """
        Validate audio quality for a segment
        
        Args:
            segment: Audio segment data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check duration
        logger.debug(f"Checking duration: {segment.duration:.2f}s (min: {self.min_duration}s, max: {self.max_duration}s)")
        if segment.duration < self.min_duration:
            issues.append(f"Duration too short: {segment.duration:.2f}s < {self.min_duration}s")
        elif segment.duration > self.max_duration:
            issues.append(f"Duration too long: {segment.duration:.2f}s > {self.max_duration}s")
        
        # Check confidence
        if segment.confidence < self.min_confidence:
            issues.append(f"Low confidence: {segment.confidence:.2f} < {self.min_confidence}")
        
        # Check quality score
        if segment.quality_score and segment.quality_score < 0.5:
            issues.append(f"Low quality score: {segment.quality_score:.2f} < 0.5")
        
        # Check if audio file exists
        if not os.path.exists(segment.audio_file):
            issues.append(f"Audio file not found: {segment.audio_file}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_text_quality(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate text quality for a segment
        
        Args:
            text: Text content
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check text length
        if len(text.strip()) < self.min_text_length:
            issues.append(f"Text too short: {len(text)} chars < {self.min_text_length}")
        elif len(text.strip()) > self.max_text_length:
            issues.append(f"Text too long: {len(text)} chars > {self.max_text_length}")
        
        # Check for empty or whitespace-only text
        if not text.strip():
            issues.append("Empty or whitespace-only text")
        
        # Check for noise patterns
        for pattern in self.noise_patterns:
            if re.search(pattern, text):
                issues.append(f"Contains noise pattern: {pattern}")
        
        # Check for excessive filler words
        words = text.lower().split()
        filler_count = sum(1 for word in words if word in self.filler_words)
        filler_ratio = filler_count / len(words) if words else 0
        
        if filler_ratio > 0.3:
            issues.append(f"Too many filler words: {filler_ratio:.2f} > 0.3")
        
        # Check for repetitive text
        if self._is_repetitive(text):
            issues.append("Text appears to be repetitive")
        
        # Check for proper sentence structure
        if not self._has_proper_structure(text):
            issues.append("Poor sentence structure")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _is_repetitive(self, text: str) -> bool:
        """
        Check if text is repetitive
        
        Args:
            text: Text to check
            
        Returns:
            True if text is repetitive
        """
        words = text.lower().split()
        if len(words) < 6:
            return False
        
        # Check for repeated phrases
        for i in range(len(words) - 3):
            phrase = ' '.join(words[i:i+3])
            if text.lower().count(phrase) > 1:
                return True
        
        return False
    
    def _has_proper_structure(self, text: str) -> bool:
        """
        Check if text has proper sentence structure
        
        Args:
            text: Text to check
            
        Returns:
            True if text has proper structure
        """
        # Check for capitalization
        if not any(c.isupper() for c in text):
            return False
        
        # Check for punctuation
        if not any(c in text for c in '.!?'):
            return False
        
        # Check for reasonable word distribution
        words = text.split()
        if len(words) < 3:
            return False
        
        # Check average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < 2 or avg_word_length > 15:
            return False
        
        return True
    
    def validate_speaker_consistency(self, segments: List[AudioSegmentData]) -> Tuple[bool, List[str]]:
        """
        Validate speaker consistency across segments
        
        Args:
            segments: List of audio segments
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Group segments by speaker
        speaker_segments = {}
        for segment in segments:
            if segment.speaker not in speaker_segments:
                speaker_segments[segment.speaker] = []
            speaker_segments[segment.speaker].append(segment)
        
        # Check minimum segments per speaker
        for speaker, speaker_segs in speaker_segments.items():
            if len(speaker_segs) < 5:
                issues.append(f"Speaker {speaker} has too few segments: {len(speaker_segs)} < 5")
            
            # Check minimum duration per speaker
            total_duration = sum(seg.duration for seg in speaker_segs)
            if total_duration < 30.0:  # 30 seconds minimum
                issues.append(f"Speaker {speaker} has insufficient duration: {total_duration:.2f}s < 30s")
        
        # Check for too many speakers
        if len(speaker_segments) > 10:
            issues.append(f"Too many speakers detected: {len(speaker_segments)} > 10")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def filter_segments(self, segments: List[AudioSegmentData]) -> Tuple[List[AudioSegmentData], QualityReport]:
        """
        Filter segments based on quality criteria
        
        Args:
            segments: List of audio segments
            
        Returns:
            Tuple of (filtered_segments, quality_report)
        """
        logger.info(f"Starting quality filtering for {len(segments)} segments")
        
        passed_segments = []
        failed_segments = []
        quality_issues = {
            "audio_quality": [],
            "text_quality": [],
            "speaker_consistency": []
        }
        
        # Filter individual segments
        for segment in segments:
            try:
                logger.info(f"Filtering segment: duration={segment.duration:.2f}s, confidence={segment.confidence:.2f}, text_length={len(segment.text)}")
                audio_valid, audio_issues = self.validate_audio_quality(segment)
                text_valid, text_issues = self.validate_text_quality(segment.text)
                
                logger.info(f"Audio valid: {audio_valid}, issues: {audio_issues}")
                logger.info(f"Text valid: {text_valid}, issues: {text_issues}")
                
                if audio_valid and text_valid:
                    passed_segments.append(segment)
                else:
                    failed_segments.append(segment)
                    quality_issues["audio_quality"].extend(audio_issues)
                    quality_issues["text_quality"].extend(text_issues)
            except Exception as e:
                logger.error(f"Error filtering segment: {e}")
                failed_segments.append(segment)
                quality_issues["audio_quality"].append(f"Error: {e}")
        
        # Validate speaker consistency
        speaker_valid, speaker_issues = self.validate_speaker_consistency(passed_segments)
        if not speaker_valid:
            quality_issues["speaker_consistency"].extend(speaker_issues)
        
        # Calculate statistics
        statistics = self._calculate_statistics(passed_segments, failed_segments)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(statistics, quality_issues)
        
        # Create quality report
        report = QualityReport(
            total_segments=len(segments),
            passed_segments=len(passed_segments),
            failed_segments=len(failed_segments),
            quality_issues=quality_issues,
            statistics=statistics,
            recommendations=recommendations
        )
        
        logger.info(f"Quality filtering completed: {len(passed_segments)}/{len(segments)} segments passed")
        
        return passed_segments, report
    
    def _calculate_statistics(self, passed_segments: List[AudioSegmentData], 
                            failed_segments: List[AudioSegmentData]) -> Dict[str, Any]:
        """
        Calculate quality statistics
        
        Args:
            passed_segments: Segments that passed quality checks
            failed_segments: Segments that failed quality checks
            
        Returns:
            Dictionary with statistics
        """
        all_segments = passed_segments + failed_segments
        
        if not all_segments:
            return {}
        
        # Duration statistics
        durations = [seg.duration for seg in all_segments]
        confidences = [seg.confidence for seg in all_segments]
        quality_scores = [seg.quality_score or 0.0 for seg in all_segments]
        text_lengths = [len(seg.text) for seg in all_segments]
        
        # Speaker statistics
        speakers = set(seg.speaker for seg in all_segments)
        speaker_counts = {}
        for seg in all_segments:
            speaker_counts[seg.speaker] = speaker_counts.get(seg.speaker, 0) + 1
        
        statistics = {
            "duration_stats": {
                "min": min(durations),
                "max": max(durations),
                "mean": sum(durations) / len(durations),
                "median": sorted(durations)[len(durations) // 2]
            },
            "confidence_stats": {
                "min": min(confidences),
                "max": max(confidences),
                "mean": sum(confidences) / len(confidences),
                "median": sorted(confidences)[len(confidences) // 2]
            },
            "quality_score_stats": {
                "min": min(quality_scores),
                "max": max(quality_scores),
                "mean": sum(quality_scores) / len(quality_scores),
                "median": sorted(quality_scores)[len(quality_scores) // 2]
            },
            "text_length_stats": {
                "min": min(text_lengths),
                "max": max(text_lengths),
                "mean": sum(text_lengths) / len(text_lengths),
                "median": sorted(text_lengths)[len(text_lengths) // 2]
            },
            "speaker_stats": {
                "total_speakers": len(speakers),
                "speaker_distribution": speaker_counts,
                "avg_segments_per_speaker": len(all_segments) / len(speakers) if speakers else 0
            },
            "pass_rate": len(passed_segments) / len(all_segments) if all_segments else 0
        }
        
        return statistics
    
    def _generate_recommendations(self, statistics: Dict[str, Any], 
                                quality_issues: Dict[str, List[str]]) -> List[str]:
        """
        Generate recommendations based on quality analysis
        
        Args:
            statistics: Quality statistics
            quality_issues: Quality issues found
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check pass rate
        pass_rate = statistics.get("pass_rate", 0)
        if pass_rate < 0.7:
            recommendations.append("Low pass rate detected. Consider relaxing quality criteria or improving source audio quality.")
        
        # Check speaker distribution
        speaker_stats = statistics.get("speaker_stats", {})
        total_speakers = speaker_stats.get("total_speakers", 0)
        if total_speakers < 2:
            recommendations.append("Only one speaker detected. Consider adding more diverse audio sources.")
        elif total_speakers > 5:
            recommendations.append("Many speakers detected. Consider focusing on fewer speakers for better TTS quality.")
        
        # Check duration distribution
        duration_stats = statistics.get("duration_stats", {})
        mean_duration = duration_stats.get("mean", 0)
        if mean_duration < 3.0:
            recommendations.append("Average segment duration is short. Consider merging short segments or using longer audio sources.")
        elif mean_duration > 15.0:
            recommendations.append("Average segment duration is long. Consider splitting long segments for better training.")
        
        # Check confidence distribution
        confidence_stats = statistics.get("confidence_stats", {})
        mean_confidence = confidence_stats.get("mean", 0)
        if mean_confidence < 0.8:
            recommendations.append("Low average confidence. Consider using higher quality audio sources or adjusting transcription settings.")
        
        # Check quality score distribution
        quality_stats = statistics.get("quality_score_stats", {})
        mean_quality = quality_stats.get("mean", 0)
        if mean_quality < 0.7:
            recommendations.append("Low average quality score. Consider improving audio preprocessing or source quality.")
        
        # Check for specific issues
        if quality_issues.get("audio_quality"):
            recommendations.append("Audio quality issues detected. Review audio preprocessing pipeline.")
        
        if quality_issues.get("text_quality"):
            recommendations.append("Text quality issues detected. Consider improving transcription settings or post-processing.")
        
        if quality_issues.get("speaker_consistency"):
            recommendations.append("Speaker consistency issues detected. Review speaker diarization settings.")
        
        return recommendations
    
    def generate_quality_report(self, segmentation_result: SegmentationResult) -> QualityReport:
        """
        Generate comprehensive quality report for segmentation result
        
        Args:
            segmentation_result: Segmentation result
            
        Returns:
            QualityReport object
        """
        logger.info("Generating quality report")
        
        # Filter segments
        filtered_segments, report = self.filter_segments(segmentation_result.segments)
        
        # Add additional analysis
        report.statistics.update({
            "original_segments": len(segmentation_result.segments),
            "filtered_segments": len(filtered_segments),
            "filter_rate": len(filtered_segments) / len(segmentation_result.segments) if segmentation_result.segments else 0,
            "total_duration": sum(seg.duration for seg in segmentation_result.segments),
            "filtered_duration": sum(seg.duration for seg in filtered_segments)
        })
        
        return report
    
    def save_quality_report(self, report: QualityReport, output_path: str = None) -> str:
        """
        Save quality report to file
        
        Args:
            report: Quality report
            output_path: Output path (optional)
            
        Returns:
            Path to saved report
        """
        try:
            if output_path is None:
                output_path = os.path.join(settings.output_dir, "metadata", "quality_report.json")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as JSON
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved quality report to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save quality report: {e}")
            raise
