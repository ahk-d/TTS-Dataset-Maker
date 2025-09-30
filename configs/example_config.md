# Dataset Configuration Example

This file shows the structure for configuring your TTS dataset processing.

## Example Configuration

### Simple Format (Auto-transcription)
```json
{
  "name": "my_dataset",
  "sources": [
    "/path/to/audio1.mp3",
    "/path/to/audio2.wav",
    "/path/to/audio3.m4a"
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
    },
    "/path/to/audio3.m4a"
  ]
}
```

## Configuration Fields

- **name**: Name of your dataset (used for output directory)
- **sources**: Array of audio file paths or objects with file and transcript paths

### Source Object Format
- **file**: Path to the audio file
- **transcript**: Path to existing transcript JSON file (optional)

## Transcript Format

Existing transcripts should be in JSON format with this structure:
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

## Usage

1. Copy this example to `configs/dataset_config.json`
2. Update the name and sources fields with your data
3. Run: `uv run python scripts/process.py configs/dataset_config.json`

## Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)

## Example for Colab

```json
{
  "name": "colab_dataset",
  "sources": [
    {
      "file": "/content/tts-dataset-maker/my_audio.mp3",
      "transcript": "/content/tts-dataset-maker/my_transcript.json"
    }
  ]
}
```