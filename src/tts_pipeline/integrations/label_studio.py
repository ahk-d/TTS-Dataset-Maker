"""
Simple Label Studio integration for manual annotation
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path

try:
    from label_studio_sdk import Client
    LABEL_STUDIO_AVAILABLE = True
except ImportError:
    LABEL_STUDIO_AVAILABLE = False

from ..config.settings import settings

logger = logging.getLogger(__name__)


class LabelStudioManager:
    """Simple Label Studio manager for manual annotation"""
    
    def __init__(self):
        if not LABEL_STUDIO_AVAILABLE:
            logger.warning("Label Studio SDK not available. Install with: pip install label-studio-sdk")
            self.client = None
            return
        
        self.url = settings.label_studio_url
        self.api_key = settings.label_studio_api_key
        
        if self.api_key:
            self.client = Client(url=self.url, api_key=self.api_key)
        else:
            logger.warning("Label Studio API key not configured")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Label Studio is available"""
        return LABEL_STUDIO_AVAILABLE and self.client is not None
    
    def create_project(self, project_name: str, segments: List[Dict]) -> Optional[str]:
        """Create a Label Studio project for manual annotation"""
        if not self.is_available():
            logger.warning("Label Studio not available")
            return None
        
        try:
            # Create project
            project = self.client.create_project(
                title=project_name,
                description=f"TTS Dataset Annotation - {len(segments)} segments"
            )
            
            logger.info(f"Created Label Studio project: {project.id}")
            
            # Create tasks for each segment
            tasks = []
            for segment in segments:
                task_data = {
                    "data": {
                        "segment_id": segment["segment_id"],
                        "text": segment["text"],
                        "speaker_id": segment["speaker_id"],
                        "duration": segment["duration"],
                        "confidence": segment["confidence"],
                        "audio_file": segment.get("audio_file", "")
                    }
                }
                tasks.append(task_data)
            
            # Import tasks
            self.client.import_tasks(project.id, tasks)
            
            logger.info(f"Imported {len(tasks)} tasks to project {project.id}")
            logger.info(f"Label Studio URL: {self.url}")
            logger.info(f"Project URL: {self.url}/projects/{project.id}")
            
            return str(project.id)
            
        except Exception as e:
            logger.error(f"Failed to create Label Studio project: {e}")
            return None
    
    def get_annotations(self, project_id: str) -> List[Dict]:
        """Get annotations from a Label Studio project"""
        if not self.is_available():
            logger.warning("Label Studio not available")
            return []
        
        try:
            project = self.client.get_project(project_id)
            tasks = project.get_tasks()
            
            annotations = []
            for task in tasks:
                if task.annotations:
                    annotation = {
                        "task_id": task.id,
                        "segment_id": task.data.get("segment_id"),
                        "annotations": task.annotations
                    }
                    annotations.append(annotation)
            
            return annotations
            
        except Exception as e:
            logger.error(f"Failed to get annotations: {e}")
            return []
