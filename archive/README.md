# TTS Dataset Maker

**Production-ready TTS dataset creation with speaker diarization and intelligent storage**

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export ASSEMBLYAI_API_KEY="your_key_here"

# Process multiple audio files
python tts_dataset_maker.py file1.mp3 file2.mp3 file3.mp3

# Export dataset
python dataset_manager.py export my_dataset --format huggingface
```

## 📁 Core Files

### **Essential Components**
- `tts_service.py` - Main TTS service (multi-file first)
- `tts_dataset_maker.py` - Simple wrapper for easy usage
- `dataset_manager.py` - Production CLI for dataset management
- `production_storage.py` - Intelligent storage with SQLite database

### **Processing Components**
- `audio_processor.py` - Audio preprocessing and validation
- `assemblyai_client.py` - Transcription and speaker diarization
- `audio_segmenter.py` - Audio segmentation
- `metadata_generator.py` - Metadata generation
- `quality_control.py` - Quality filtering
- `voice_activity_detector.py` - Voice activity detection
- `speaker_alignment.py` - Speaker ID alignment across files

### **Configuration**
- `config.py` - Settings and configuration
- `requirements.txt` - Dependencies
- `install_dependencies.py` - Dependency installer

## 🎯 Usage

### **Simple Processing**
```bash
# Process files
python tts_dataset_maker.py *.mp3 --output ./my_dataset

# Check status
python dataset_manager.py status

# Export dataset
python dataset_manager.py export my_dataset
```

### **Advanced Management**
```bash
# List files
python dataset_manager.py list-files

# List segments
python dataset_manager.py list-segments

# List speakers
python dataset_manager.py list-speakers

# Validate dataset
python dataset_manager.py validate
```

### **Speaker Alignment**
```bash
# Option 1: Web UI (Recommended)
python dataset_manager.py ui

# Option 2: Command Line
# Step 1: Create speaker reference guide
python dataset_manager.py create-reference

# Step 2: Create alignment workflow
python dataset_manager.py create-workflow

# Step 3: Analyze speakers across files
python dataset_manager.py analyze-speakers

# Step 4: Apply alignment (dry run first)
python dataset_manager.py apply-alignment alignment_file.json --dry-run

# Step 5: Apply alignment
python dataset_manager.py apply-alignment alignment_file.json

# Step 6: Verify alignment results
python dataset_manager.py verify-alignment alignment_file.json
```

### **Python API**
```python
from tts_service import TTSDatasetService

# Initialize service
service = TTSDatasetService("./output")

# Process files
dataset_metadata, stats = service.run_complete_pipeline([
    "file1.mp3", "file2.mp3", "file3.mp3"
])

# Get statistics
stats = service.get_dataset_statistics()
print(f"Total segments: {stats['audio_segments']['total']}")
```

## 🏗️ Architecture

### **Multi-File First Design**
- Single file processing is a special case of multi-file processing
- Intelligent storage with SQLite database
- Source file tracking and deduplication
- Comprehensive metadata management

### **Storage Structure**
```
output/
├── dataset.db              # SQLite database
├── source_files/           # Original files
├── audio_segments/         # Processed segments
├── metadata/               # Legacy metadata
├── temp/                  # Temporary files
└── exports/               # Dataset exports
```

## 🔧 Configuration

### **Environment Variables**
```bash
export ASSEMBLYAI_API_KEY="your_key_here"
export OUTPUT_DIR="./output"
export BATCH_SIZE=1
export SPEAKER_MAPPING_MODE="interactive"
```

### **Settings**
```python
# config.py
class Settings:
    # Multi-file processing
    batch_size: int = 1
    speaker_mapping_mode: str = "interactive"
    
    # Quality control
    min_segment_duration: float = 1.0
    min_confidence_score: float = 0.7
    
    # VAD settings
    vad_enabled: bool = True
    min_speech_duration: float = 0.5
```

## 📊 Features

### **Production-Ready**
- ✅ SQLite database storage
- ✅ File deduplication (SHA256)
- ✅ Source tracking and metadata
- ✅ Error handling and recovery
- ✅ Dataset validation
- ✅ Multiple export formats

### **Multi-File Processing**
- ✅ Process 1 or 1000 files the same way
- ✅ Interactive speaker mapping
- ✅ Batch processing
- ✅ Quality filtering
- ✅ Comprehensive statistics

### **Export Formats**
- ✅ Hugging Face format
- ✅ Orpheus TTS format
- ✅ Custom metadata preservation
- ✅ Dataset cards and documentation

## 🎯 Best Practices

1. **Use the simple wrapper** (`tts_dataset_maker.py`) for most cases
2. **Check dataset status** before processing large batches
3. **Use speaker alignment workflow** for consistent speaker IDs
4. **Validate dataset** after processing
5. **Export regularly** to preserve your work

## 🎤 Speaker Alignment Workflow

### **The Problem**
Diarization models assign arbitrary labels (A, B, C) that don't correspond to actual speakers. For example:
- **File 1**: Speaker A = John, Speaker B = Mary
- **File 2**: Speaker A = Mary, Speaker B = John

### **The Solution**

#### **Option 1: Web UI (Recommended)**
```bash
# Start the web annotation interface
python dataset_manager.py ui
```
- **Dashboard**: Overview of all files and statistics
- **Audio Playback**: Listen to segments directly in the browser
- **Visual Interface**: Drag-and-drop speaker alignment
- **Real-time Preview**: See alignment changes immediately
- **Production-grade**: Based on industry tools like Label Studio

#### **Option 2: Command Line**
1. **Create Reference**: `python dataset_manager.py create-reference`
   - Shows all speakers with sample segments
   - Helps identify which segments belong to the same person

2. **Create Workflow**: `python dataset_manager.py create-workflow`
   - Creates a step-by-step alignment guide
   - Shows sample segments for each file

3. **Manual Alignment**: Edit the workflow file to map speakers
   ```json
   {
     "speaker_alignments": {
       "file1_A": "speaker_john",
       "file1_B": "speaker_mary", 
       "file2_A": "speaker_mary",
       "file2_B": "speaker_john"
     }
   }
   ```

4. **Apply Alignment**: Test and apply the alignment
   ```bash
   # Test first
   python dataset_manager.py apply-alignment workflow.json --dry-run
   
   # Apply alignment
   python dataset_manager.py apply-alignment workflow.json
   ```

5. **Verify Results**: Check the final speaker distribution
   ```bash
   python dataset_manager.py verify-alignment workflow.json
   ```

## 🚨 Requirements

- Python 3.10+
- AssemblyAI API key
- 8GB+ RAM for large datasets
- SSD storage recommended

## 📈 Production Checklist

- [ ] Set `ASSEMBLYAI_API_KEY` environment variable
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Test with small dataset first
- [ ] Monitor disk space for large datasets
- [ ] Set up regular backups
- [ ] Configure logging and monitoring

## 🎉 Benefits

- **Scalable**: Handles any number of files
- **Reliable**: Production-ready storage system
- **Trackable**: Full source and metadata tracking
- **Exportable**: Multiple format support
- **Manageable**: Comprehensive CLI tools
- **Validatable**: Dataset integrity checking

---

**Ready for production use!** 🚀