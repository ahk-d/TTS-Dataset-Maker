#!/usr/bin/env python3
"""
Hugging Face Dataset Utilities
Utilities for uploading TTS datasets to Hugging Face Hub
"""

import json
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def create_dataset_card(output_dir, dataset_name="TTS Dataset", description=None):
    """Create a proper dataset card with YAML metadata"""
    
    if description is None:
        description = f"This dataset contains audio segments with transcriptions for Text-to-Speech training."
    
    readme_content = f"""---
language:
- en
pretty_name: "{dataset_name}"
tags:
- audio
- speech
- tts
- transcription
- english
license: "mit"
task_categories:
- text-to-speech
- automatic-speech-recognition
size_categories:
- 1K<n<10K
---

# {dataset_name}

{description}

## Dataset Structure
- `audio_segments/`: Audio files (WAV format, 24kHz)
- `metadata/`: Contains TTS metadata JSON files
- `transcription_cache/`: Transcription cache files

## Usage
```python
import json
import os

# Load metadata
metadata_dir = "metadata"
for file in os.listdir(metadata_dir):
    if file.startswith("tts_metadata_") and file.endswith(".json"):
        with open(os.path.join(metadata_dir, file), 'r') as f:
            data = json.load(f)
        break

# Access segments
for segment in data:
    print(f"Text: {{segment['text']}}")
    print(f"Speaker: {{segment['speaker_id']}}")
    print(f"Audio: {{segment['audio_file']}}")
```

## Dataset Details
- **Language**: English
- **Format**: WAV audio files (24kHz)
- **Speakers**: Multiple speakers
- **Duration**: Variable segment lengths
- **Quality**: High-quality audio with quality scores

## Metadata Format
Each segment contains:
- `text`: Transcribed text
- `speaker_id`: Speaker identifier
- `audio_file`: Path to audio file
- `duration`: Duration in seconds
- `quality_score`: Audio quality score
- `confidence_score`: Transcription confidence

## License
MIT License

## Citation
If you use this dataset, please cite:
```
@dataset{{{dataset_name.lower().replace(' ', '_')},
  title={{{dataset_name}}},
  author={{Your Name}},
  year={{2024}},
  url={{https://huggingface.co/datasets/your-username/your-dataset}}
}}
```
"""
    
    # Write README to output directory
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Dataset card created at {readme_path}")
    return readme_path

def upload_to_huggingface(source_dir, repo_name, token, private=False):
    """Upload existing output to Hugging Face"""
    
    # Create repository
    api = HfApi(token=token)
    create_repo(
        repo_id=repo_name, 
        repo_type="dataset", 
        token=token,
        private=private
    )
    
    # Upload the entire output folder
    api.upload_folder(
        folder_path=source_dir,
        repo_id=repo_name,
        repo_type="dataset",
        token=token
    )
    
    print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_name}")
    return f"https://huggingface.co/datasets/{repo_name}"

def prepare_and_upload(source_dir, repo_name, token, dataset_name=None, description=None, private=False):
    """Complete workflow: create dataset card and upload to Hugging Face"""
    
    if dataset_name is None:
        dataset_name = f"TTS Dataset - {Path(source_dir).name}"
    
    # Create dataset card
    create_dataset_card(source_dir, dataset_name, description)
    
    # Upload to Hugging Face
    url = upload_to_huggingface(source_dir, repo_name, token, private)
    
    return url

def copy_for_huggingface(source_dir, target_dir):
    """Copy existing output to new folder for Hugging Face (without touching original)"""
    
    # Create new directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy everything from output
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    
    print(f"Copied dataset to {target_dir}")
    return target_dir

def reorganize_for_huggingface(source_dir, target_dir):
    """Reorganize output for Hugging Face dataset format"""
    
    # Create new dataset directory
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(f"{target_dir}/audio", exist_ok=True)
    
    # Find and copy the TTS metadata
    metadata_dir = os.path.join(source_dir, "metadata")
    for file in os.listdir(metadata_dir):
        if file.startswith("tts_metadata_") and file.endswith(".json"):
            # Copy the TTS metadata as the main metadata file
            shutil.copy2(
                os.path.join(metadata_dir, file), 
                f"{target_dir}/metadata.json"
            )
            break
    
    # Copy audio files to audio/ subdirectory
    audio_source_dir = os.path.join(source_dir, "audio_segments")
    audio_target_dir = f"{target_dir}/audio"
    
    if os.path.exists(audio_source_dir):
        for file in os.listdir(audio_source_dir):
            if file.endswith(".wav"):
                shutil.copy2(
                    os.path.join(audio_source_dir, file),
                    os.path.join(audio_target_dir, file)
                )
    
    print(f"Dataset reorganized for Hugging Face in {target_dir}")
    return target_dir

# Example usage
if __name__ == "__main__":
    # Example 1: Upload existing output directly
    # prepare_and_upload(
    #     source_dir="/content/TTS-Dataset-Maker/output",
    #     repo_name="your-username/your-tts-dataset",
    #     token="your_huggingface_token",
    #     dataset_name="Peaky Blinders TTS Dataset",
    #     description="Audio segments with transcriptions from Peaky Blinders for TTS training"
    # )
    
    # Example 2: Copy and reorganize first
    # target_dir = copy_for_huggingface("/content/TTS-Dataset-Maker/output", "./hf_dataset")
    # prepare_and_upload(target_dir, "your-username/your-tts-dataset", "your_token")
    
    print("Hugging Face utilities ready!")
    print("Use prepare_and_upload() for direct upload")
    print("Use copy_for_huggingface() + prepare_and_upload() for safe copying")
