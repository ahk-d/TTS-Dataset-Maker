# TTS Dataset Maker

Simple local TTS dataset creation pipeline with Silero VAD-based silence removal, DeepFilterNet denoising, and Label Studio integration.

## Features

- **Silero VAD-based silence removal** - Removes long silences while preserving natural speech gaps
- **DeepFilterNet denoising** - CPU-optimized audio denoising
- **AssemblyAI transcription** - High-quality speech-to-text with speaker diarization
- **Label Studio integration** - Manual annotation and quality control
- **Hugging Face export** - Ready-to-use TTS dataset format

## Quick Start

### 1. Install Dependencies

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
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
```

### 4. Process Your Dataset

```bash
# Process a single file
python local_processor.py configs/dataset_config.json

# Or use the simple script
python scripts/process.py configs/dataset_config.json
```

## Configuration

Edit `configs/dataset_config.json`:

```json
{
  "name": "my_dataset",
  "sources": [
    "/path/to/audio1.mp3",
    "/path/to/audio2.wav"
  ]
}
```

## Environment Variables

- `ASSEMBLYAI_API_KEY` - Your AssemblyAI API key (required)
- `LABEL_STUDIO_URL` - Label Studio URL (default: http://localhost:8080)
- `LABEL_STUDIO_API_KEY` - Label Studio API key (optional)
- `VAD_ENABLED` - Enable VAD (default: true)
- `VAD_METHOD` - VAD method (default: silero)
- `REMOVE_LONG_SILENCES` - Remove long silences (default: true)
- `MAX_SILENCE_DURATION` - Max silence duration to keep (default: 1.0s)

## Output

Processed datasets are saved to `output/`:
- `audio_segments/` - Individual audio segments
- `exports/` - Hugging Face format datasets
- `label_studio.sqlite3` - Local database

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