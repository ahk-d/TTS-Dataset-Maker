#!/usr/bin/env python3
"""
Speech Annotation UI - Production-grade web interface for speaker alignment
Based on industry tools like Label Studio, Prodigy, and custom annotation platforms
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import librosa
import soundfile as sf
import numpy as np

from production_storage import ProductionStorage
from speaker_alignment import SpeakerAlignmentManager


class SpeechAnnotationUI:
    """Production-grade speech annotation interface"""
    
    def __init__(self, dataset_dir: str = "./output", host: str = "0.0.0.0", port: int = 5000):
        self.dataset_dir = Path(dataset_dir)
        self.host = host
        self.port = port
        
        # Initialize components
        self.storage = ProductionStorage(dataset_dir)
        self.alignment_manager = SpeakerAlignmentManager(dataset_dir)
        
        # Flask app
        self.app = Flask(__name__)
        self.app.secret_key = 'speech_annotation_secret_key'
        
        # Setup routes
        self._setup_routes()
        
        # Annotation session data
        self.current_session = None
        self.annotation_data = {}
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard"""
            stats = self.storage.get_dataset_statistics()
            return render_template('index.html', stats=stats)
        
        @self.app.route('/api/files')
        def api_files():
            """Get list of files"""
            files = self.storage.get_source_files("completed")
            return jsonify([{
                'file_id': f.file_id,
                'file_name': Path(f.original_path).name,
                'duration': f.duration,
                'status': f.status,
                'created_at': f.created_at.isoformat()
            } for f in files])
        
        @self.app.route('/api/segments/<file_id>')
        def api_segments(file_id):
            """Get segments for a file"""
            segments = self.storage.get_audio_segments(source_file_id=file_id)
            return jsonify([{
                'segment_id': s.segment_id,
                'text': s.text,
                'speaker_id': s.speaker_id,
                'duration': s.duration,
                'start_time': s.start_time,
                'end_time': s.end_time,
                'audio_file': s.audio_file_path,
                'quality_score': s.quality_score
            } for s in segments])
        
        @self.app.route('/api/audio/<path:audio_path>')
        def api_audio(audio_path):
            """Serve audio files"""
            return send_file(audio_path)
        
        @self.app.route('/api/speakers')
        def api_speakers():
            """Get all speakers with statistics"""
            segments = self.storage.get_audio_segments()
            speaker_stats = {}
            
            for segment in segments:
                if segment.speaker_id not in speaker_stats:
                    speaker_stats[segment.speaker_id] = {
                        'speaker_id': segment.speaker_id,
                        'total_segments': 0,
                        'total_duration': 0.0,
                        'files': set(),
                        'sample_segments': []
                    }
                
                speaker_stats[segment.speaker_id]['total_segments'] += 1
                speaker_stats[segment.speaker_id]['total_duration'] += segment.duration
                speaker_stats[segment.speaker_id]['files'].add(segment.source_file_id)
                
                # Keep first 3 segments as samples
                if len(speaker_stats[segment.speaker_id]['sample_segments']) < 3:
                    speaker_stats[segment.speaker_id]['sample_segments'].append({
                        'segment_id': segment.segment_id,
                        'text': segment.text,
                        'duration': segment.duration,
                        'audio_file': segment.audio_file_path,
                        'source_file_id': segment.source_file_id
                    })
            
            # Convert sets to lists for JSON serialization
            for stats in speaker_stats.values():
                stats['files'] = list(stats['files'])
            
            return jsonify(list(speaker_stats.values()))
        
        @self.app.route('/annotation/<file_id>')
        def annotation_page(file_id):
            """Speaker annotation page for a specific file"""
            return render_template('annotation.html', file_id=file_id)
        
        @self.app.route('/api/alignment/save', methods=['POST'])
        def api_save_alignment():
            """Save speaker alignment"""
            data = request.json
            alignment_file = self.dataset_dir / "speaker_alignments" / f"alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            alignment_data = {
                "alignment_info": {
                    "created_at": datetime.now().isoformat(),
                    "status": "draft",
                    "total_mappings": len(data.get('mappings', []))
                },
                "manual_alignments": {
                    "mappings": data.get('mappings', [])
                },
                "global_speaker_mapping": data.get('global_mapping', {}),
                "notes": data.get('notes', '')
            }
            
            # Save alignment file
            alignment_file.parent.mkdir(exist_ok=True)
            with open(alignment_file, 'w') as f:
                json.dump(alignment_data, f, indent=2)
            
            return jsonify({
                'success': True,
                'alignment_file': str(alignment_file),
                'message': 'Alignment saved successfully'
            })
        
        @self.app.route('/api/alignment/apply', methods=['POST'])
        def api_apply_alignment():
            """Apply speaker alignment"""
            data = request.json
            alignment_file = data.get('alignment_file')
            dry_run = data.get('dry_run', True)
            
            try:
                results = self.alignment_manager.apply_alignment(alignment_file, dry_run)
                return jsonify({
                    'success': True,
                    'results': results,
                    'message': 'Alignment applied successfully' if not dry_run else 'Dry run completed'
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 400
        
        @self.app.route('/api/alignment/verify', methods=['POST'])
        def api_verify_alignment():
            """Verify alignment results"""
            data = request.json
            alignment_file = data.get('alignment_file')
            
            try:
                verification = self.alignment_manager.verify_alignment(alignment_file)
                return jsonify({
                    'success': True,
                    'verification': verification
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 400
        
        @self.app.route('/api/audio/preview/<segment_id>')
        def api_audio_preview(segment_id):
            """Generate audio preview for a segment"""
            segments = self.storage.get_audio_segments()
            segment = next((s for s in segments if s.segment_id == segment_id), None)
            
            if not segment:
                return jsonify({'error': 'Segment not found'}), 404
            
            # Generate preview (first 10 seconds)
            try:
                audio, sr = librosa.load(segment.audio_file_path, sr=22050)
                preview_duration = min(10.0, len(audio) / sr)
                preview_audio = audio[:int(preview_duration * sr)]
                
                # Save preview
                preview_path = self.dataset_dir / "temp" / f"preview_{segment_id}.wav"
                preview_path.parent.mkdir(exist_ok=True)
                sf.write(str(preview_path), preview_audio, sr)
                
                return jsonify({
                    'preview_path': str(preview_path),
                    'duration': preview_duration
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def create_templates(self):
        """Create HTML templates for the UI"""
        
        templates_dir = self.dataset_dir / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        # Main dashboard template
        index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speech Annotation Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stat-value { font-size: 2em; font-weight: bold; color: #3498db; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }
        .files-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .file-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .file-name { font-weight: bold; margin-bottom: 10px; }
        .file-info { color: #7f8c8d; font-size: 0.9em; }
        .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
        .btn:hover { background: #2980b9; }
        .btn-success { background: #27ae60; }
        .btn-warning { background: #f39c12; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 Speech Annotation Dashboard</h1>
            <p>Production-grade speaker alignment and annotation</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="total-files">-</div>
                <div class="stat-label">Source Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="total-segments">-</div>
                <div class="stat-label">Audio Segments</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="total-speakers">-</div>
                <div class="stat-label">Unique Speakers</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="total-duration">-</div>
                <div class="stat-label">Total Duration</div>
            </div>
        </div>
        
        <div class="files-grid" id="files-grid">
            <!-- Files will be loaded here -->
        </div>
    </div>
    
    <script>
        // Load dashboard data
        fetch('/api/files')
            .then(response => response.json())
            .then(files => {
                const filesGrid = document.getElementById('files-grid');
                files.forEach(file => {
                    const fileCard = document.createElement('div');
                    fileCard.className = 'file-card';
                    fileCard.innerHTML = `
                        <div class="file-name">${file.file_name}</div>
                        <div class="file-info">
                            Duration: ${file.duration.toFixed(1)}s<br>
                            Status: ${file.status}<br>
                            Created: ${new Date(file.created_at).toLocaleDateString()}
                        </div>
                        <a href="/annotation/${file.file_id}" class="btn">Annotate Speakers</a>
                    `;
                    filesGrid.appendChild(fileCard);
                });
            });
        
        // Load statistics
        fetch('/api/speakers')
            .then(response => response.json())
            .then(speakers => {
                document.getElementById('total-speakers').textContent = speakers.length;
                document.getElementById('total-segments').textContent = speakers.reduce((sum, s) => sum + s.total_segments, 0);
                document.getElementById('total-duration').textContent = (speakers.reduce((sum, s) => sum + s.total_duration, 0) / 60).toFixed(1) + 'm';
            });
    </script>
</body>
</html>
        """
        
        with open(templates_dir / "index.html", 'w') as f:
            f.write(index_html)
        
        # Annotation page template
        annotation_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speaker Annotation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .speakers-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .speaker-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .speaker-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .speaker-id { font-size: 1.2em; font-weight: bold; color: #2c3e50; }
        .speaker-stats { color: #7f8c8d; font-size: 0.9em; }
        .segments { margin-top: 15px; }
        .segment { border: 1px solid #ddd; border-radius: 4px; padding: 10px; margin-bottom: 10px; background: #f9f9f9; }
        .segment-text { margin-bottom: 8px; }
        .segment-controls { display: flex; gap: 10px; align-items: center; }
        .btn { background: #3498db; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #2980b9; }
        .btn-success { background: #27ae60; }
        .btn-warning { background: #f39c12; }
        .btn-danger { background: #e74c3c; }
        .alignment-controls { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .alignment-mapping { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 10px; }
        .mapping-input { padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        .save-btn { background: #27ae60; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 1.1em; }
        .save-btn:hover { background: #229954; }
        audio { width: 100%; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 Speaker Annotation</h1>
            <p>File ID: <span id="file-id"></span></p>
            <a href="/" class="btn">← Back to Dashboard</a>
        </div>
        
        <div class="alignment-controls">
            <h3>Speaker Alignment</h3>
            <div id="alignment-mappings">
                <!-- Alignment mappings will be loaded here -->
            </div>
            <button class="save-btn" onclick="saveAlignment()">Save Alignment</button>
        </div>
        
        <div class="speakers-grid" id="speakers-grid">
            <!-- Speakers will be loaded here -->
        </div>
    </div>
    
    <script>
        const fileId = window.location.pathname.split('/').pop();
        document.getElementById('file-id').textContent = fileId;
        
        // Load segments for this file
        fetch(`/api/segments/${fileId}`)
            .then(response => response.json())
            .then(segments => {
                // Group segments by speaker
                const speakerGroups = {};
                segments.forEach(segment => {
                    if (!speakerGroups[segment.speaker_id]) {
                        speakerGroups[segment.speaker_id] = [];
                    }
                    speakerGroups[segment.speaker_id].push(segment);
                });
                
                // Create speaker cards
                const speakersGrid = document.getElementById('speakers-grid');
                Object.keys(speakerGroups).forEach(speakerId => {
                    const segments = speakerGroups[speakerId];
                    const speakerCard = document.createElement('div');
                    speakerCard.className = 'speaker-card';
                    speakerCard.innerHTML = `
                        <div class="speaker-header">
                            <div class="speaker-id">Speaker ${speakerId}</div>
                            <div class="speaker-stats">${segments.length} segments</div>
                        </div>
                        <div class="segments">
                            ${segments.map(segment => `
                                <div class="segment">
                                    <div class="segment-text">"${segment.text}"</div>
                                    <div class="segment-controls">
                                        <audio controls>
                                            <source src="/api/audio/${segment.audio_file}" type="audio/wav">
                                        </audio>
                                        <span>Duration: ${segment.duration.toFixed(1)}s</span>
                                        <span>Quality: ${(segment.quality_score * 100).toFixed(1)}%</span>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    `;
                    speakersGrid.appendChild(speakerCard);
                });
                
                // Create alignment mappings
                createAlignmentMappings(Object.keys(speakerGroups));
            });
        
        function createAlignmentMappings(speakers) {
            const mappingsContainer = document.getElementById('alignment-mappings');
            speakers.forEach(speakerId => {
                const mappingDiv = document.createElement('div');
                mappingDiv.className = 'alignment-mapping';
                mappingDiv.innerHTML = `
                    <input type="text" value="Speaker ${speakerId}" readonly class="mapping-input">
                    <span>→</span>
                    <input type="text" placeholder="Enter aligned speaker ID" class="mapping-input" data-speaker="${speakerId}">
                `;
                mappingsContainer.appendChild(mappingDiv);
            });
        }
        
        function saveAlignment() {
            const mappings = [];
            const mappingInputs = document.querySelectorAll('input[data-speaker]');
            
            mappingInputs.forEach(input => {
                if (input.value.trim()) {
                    mappings.push({
                        file_id: fileId,
                        original_speaker_id: input.dataset.speaker,
                        aligned_speaker_id: input.value.trim(),
                        confidence: 1.0,
                        notes: 'Manual alignment via UI'
                    });
                }
            });
            
            fetch('/api/alignment/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mappings })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Alignment saved successfully!');
                } else {
                    alert('Error saving alignment: ' + data.message);
                }
            });
        }
    </script>
</body>
</html>
        """
        
        with open(templates_dir / "annotation.html", 'w') as f:
            f.write(annotation_html)
    
    def run(self, debug: bool = False):
        """Run the annotation UI server"""
        
        print(f"🎤 Starting Speech Annotation UI")
        print(f"📁 Dataset directory: {self.dataset_dir}")
        print(f"🌐 Server: http://{self.host}:{self.port}")
        print(f"📊 Dashboard: http://{self.host}:{self.port}/")
        
        # Create templates
        self.create_templates()
        
        # Run Flask app
        self.app.run(host=self.host, port=self.port, debug=debug)


def main():
    """Main entry point for the annotation UI"""
    parser = argparse.ArgumentParser(description="Speech Annotation UI")
    parser.add_argument("--dataset-dir", "-d", default="./output", help="Dataset directory")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run UI
    ui = SpeechAnnotationUI(args.dataset_dir, args.host, args.port)
    ui.run(args.debug)


if __name__ == "__main__":
    main()
