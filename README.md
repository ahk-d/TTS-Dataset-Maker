# TTS Dataset Maker

Simple local TTS dataset creation pipeline with Silero VAD-based silence removal, DeepFilterNet denoising, and Label Studio integration.

## Features

- **Silero VAD-based silence removal** - Removes long silences while preserving natural speech gaps
- **DeepFilterNet denoising** - CPU-optimized audio denoising (enabled by default)
- **AssemblyAI transcription** - High-quality speech-to-text with speaker diarization
- **Existing transcript support** - Use pre-existing transcripts to skip transcription
- **Label Studio integration** - Manual annotation and quality control
- **Precise segmentation** - Exact audio segment extraction matching transcript timings
- **Unique file naming** - Source-aware segment names to prevent collisions

## Quick Start

### 1. Install Dependencies

#### Local development (uv recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

#### Google Colab / plain Python environments

```python
# Option A: use uv for dependency management
!pip install uv
!uv pip install -r requirements.txt

# Option B: plain pip (if you prefer)
!pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env with your API keys
# ASSEMBLYAI_API_KEY=your_key_here
# LABEL_STUDIO_API_KEY=your_key_here (optional)
```

### 3. Run Label Studio (Optional)

```bash
# Install Label Studio
pip install label-studio

# Start Label Studio
label-studio start

# Start static file server for audio files (in another terminal)
python3 scripts/serve_static.py dataset 8000
```

### 4. Process Your Dataset

```bash
# Create your config file first
# Look at configs/example_config.md and create configs/dataset_config.json

# Basic processing (with VAD and silence removal)
uv run python scripts/process.py configs/dataset_config.json
# or
python scripts/process.py configs/dataset_config.json

# Process without VAD (keep all audio)
VAD_ENABLED=false uv run python scripts/process.py configs/dataset_config.json
# or
VAD_ENABLED=false python scripts/process.py configs/dataset_config.json

# Process without silence removal
REMOVE_LONG_SILENCES=false uv run python scripts/process.py configs/dataset_config.json
# or
REMOVE_LONG_SILENCES=false python scripts/process.py configs/dataset_config.json

# Process with custom settings
MIN_SEGMENT_DURATION=2.0 MAX_SILENCE_DURATION=0.5 uv run python scripts/process.py configs/dataset_config.json
# or
MIN_SEGMENT_DURATION=2.0 MAX_SILENCE_DURATION=0.5 python scripts/process.py configs/dataset_config.json
```

## Configuration

Create your config file:

1. **Look at the example**: Open `configs/example_config.md` to see the JSON structure
2. **Create your config**: Create a new file `configs/dataset_config.json` with your settings
3. **Edit the paths**: Update the `sources` array with your audio file paths

### Simple Format (Auto-transcription)
```json
{
  "name": "my_dataset",
  "sources": [
    "/path/to/audio1.mp3",
    "/path/to/audio2.wav"
  ]
}
```

### Advanced Format (With Existing Transcripts)
```json
{
  "name": "my_dataset",
  "sources": [
    {
      "file": "/path/to/audio1.mp3",
      "transcript": "/path/to/transcript1.json"
    },
    {
      "file": "/path/to/audio2.wav",
      "transcript": "/path/to/transcript2.json"
    }
  ]
}
```

### Transcript Format
If you have existing transcripts, they should be JSON files with this structure:
```json
{
  "utterances": [
    {
      "text": "Hello world",
      "speaker": "A",
      "start": 0.0,
      "end": 1.5,
      "confidence": 0.95
    }
  ]
}
```

## Environment Variables

### Required
- `ASSEMBLYAI_API_KEY` - Your AssemblyAI API key (required)

### Optional
- `LABEL_STUDIO_URL` - Label Studio URL (default: http://localhost:8080)
- `LABEL_STUDIO_API_KEY` - Label Studio API key (optional)

### Audio Processing Settings
- `VAD_ENABLED` - Enable Voice Activity Detection (default: true)
- `VAD_METHOD` - VAD method: silero, webrtc (default: silero)
- `REMOVE_LONG_SILENCES` - Remove long silences (default: true)
- `MAX_SILENCE_DURATION` - Max silence duration to keep (default: 1.0s)
- `SILENCE_THRESHOLD` - Silence detection threshold in dB (default: -30.0)
- `MIN_SPEECH_DURATION` - Minimum speech segment duration (default: 0.5s)
- `MIN_SPEECH_RATIO` - Minimum speech ratio in segment (default: 0.3)

### Quality Control Settings
- `MIN_SEGMENT_DURATION` - Minimum segment duration (default: 1.0s)
- `MAX_SEGMENT_DURATION` - Maximum segment duration (default: 3600.0s)
- `MIN_CONFIDENCE_SCORE` - Minimum transcription confidence (default: 0.7)

## Output

Processed datasets are saved to `dataset/`:
- `audio/` - Individual audio segments with unique names (e.g., `pb1_segment_0.wav`)
- `metadata.json` - Complete segment metadata for Label Studio
- `transcripts/` - Per-file transcript JSON files

## Dependencies

- Python 3.10+
- PyTorch (CPU)
- Silero VAD
- DeepFilterNet
- AssemblyAI
- Label Studio SDK
- Librosa
- SoundFile

## License

MIT