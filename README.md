# TTS Dataset Maker

A modular tool for creating and exploring TTS (Text-to-Speech) datasets from YouTube videos using AssemblyAI.

## File Structure

```
TTS-Dataset-Maker/
├── main.py                    # Main entry point for the Gradio interface
├── data_processor.py          # Handles JSON and audio file operations
├── ui_components.py           # Gradio interface components and event handlers
├── youtube_processor.py       # YouTube video processing utilities
├── metadata_generator.py      # Generates metadata and extracts audio segments
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── tts_dataset_maker.py      # Original script (for reference)
```

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd TTS-Dataset-Maker
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Process a YouTube Video

First, you need to process a YouTube video to create the dataset:

```bash
python youtube_processor.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --assemblyai-key "YOUR_ASSEMBLYAI_KEY"
```

This will:
- Download the YouTube video
- Process it with AssemblyAI for transcription and speaker diarization
- Save the results to `output/tts_dataset.json` and `output/audio.wav`

### 2. Launch the Explorer Interface

After processing the video, launch the Gradio interface:

```bash
python main.py
```

This will:
- Load the processed data
- Launch a web interface for exploring the segments
- Allow you to filter by speaker, search text, and play audio segments

### 3. Generate Metadata (Optional)

To create individual audio segments and metadata for TTS training:

```bash
python metadata_generator.py
```

This will:
- Extract individual audio segments for each speaker
- Generate `metadata_raw.json` and `metadata.json` files
- Create a structured dataset ready for TTS training

## Features

### Data Processing (`data_processor.py`)
- Loads and validates JSON and audio files
- Processes segments into a pandas DataFrame
- Handles audio slicing and concatenation
- Provides filtering and search functionality

### UI Components (`ui_components.py`)
- Creates the Gradio web interface
- Handles user interactions (filtering, row selection)
- Manages audio playback and segment exploration

### YouTube Processing (`youtube_processor.py`)
- Processes YouTube videos with AssemblyAI
- Handles command-line arguments

### Metadata Generation (`metadata_generator.py`)
- Extracts individual audio segments for each speaker
- Generates `metadata_raw.json` and `metadata.json` files
- Creates structured dataset for TTS training
- Filters segments by duration and text quality
- Provides detailed statistics and speaker analysis

## Output Files

After processing, you should have:
- `output/tts_dataset.json` - Transcription and speaker data
- `output/audio.wav` - The original audio file

After running the metadata generator, you'll also have:
- `output/segments/metadata_raw.json` - Original segment data with audio paths
- `output/segments/metadata.json` - Cleaned data optimized for TTS training
- `output/segments/audio/` - Individual audio segments (segment_XXXXXX.wav)

## Interface Features

- **Speaker Filtering**: Filter segments by specific speakers
- **Text Search**: Search through transcriptions
- **Audio Playback**: Click any row to hear that exact segment
- **Concatenation**: Play all matching segments together
- **Export**: View and copy transcriptions

## Requirements

- Python 3.7+
- AssemblyAI API key
- Internet connection for YouTube downloads
- Sufficient disk space for audio files

## Troubleshooting

1. **Missing files**: Make sure you've run the YouTube processor first
2. **AssemblyAI errors**: Check your API key and quota
3. **Audio issues**: Ensure the audio file is valid and accessible
4. **Gradio issues**: Try reinstalling with `pip install gradio==5.* --force-reinstall` 