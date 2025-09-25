# Dataset Configuration Example

This file shows the structure for configuring your TTS dataset processing.

## Example Configuration

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

## Configuration Fields

- **`name`**: Name of your dataset (used for output directory)
- **`sources`**: Array of audio file paths to process

## Usage

1. Copy this example to `configs/dataset_config.json`
2. Update the `name` and `sources` fields with your data
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
    "/content/tts-dataset-maker/my_audio.mp3"
  ]
}
```
