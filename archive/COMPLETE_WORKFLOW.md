# TTS Dataset Maker - Complete Workflow Guide

**From Audio Files to Hugging Face Dataset - Complete Production Pipeline**

## 🎯 **Overview**

This guide covers the complete workflow from processing audio files to publishing a Hugging Face dataset, including speaker alignment, quality control, and production deployment.

---

## 📋 **Prerequisites**

### **System Requirements**
- Python 3.10+
- 8GB+ RAM (for large datasets)
- SSD storage (recommended)
- Internet connection (for AssemblyAI API)

### **API Keys**
```bash
# Set your AssemblyAI API key
export ASSEMBLYAI_API_KEY="your_api_key_here"

# Optional: Set custom output directory
export OUTPUT_DIR="./output"
```

### **Installation**
```bash
# Install dependencies
pip install -r requirements.txt

# Or use the installer
python install_dependencies.py
```

---

## 🚀 **Step 1: Process Audio Files**

### **Simple Processing (Recommended)**
```bash
# Process multiple audio files
python tts_dataset_maker.py file1.mp3 file2.mp3 file3.mp3

# With custom output directory
python tts_dataset_maker.py *.mp3 --output ./my_dataset --name "my_tts_dataset"

# Non-interactive mode
python tts_dataset_maker.py *.mp3 --no-interactive
```

### **Advanced Processing**
```bash
# Full control with all options
python tts_service.py file1.mp3 file2.mp3 file3.mp3 \
    --output-dir ./output \
    --dataset-name "my_tts_dataset" \
    --language en \
    --batch-size 1 \
    --speaker-mapping interactive \
    --vad \
    --verbose
```

### **What Happens During Processing**
1. **Audio Preprocessing**: Validates and preprocesses audio files
2. **Transcription**: Uses AssemblyAI for transcription and speaker diarization
3. **Segmentation**: Segments audio based on speaker changes
4. **Quality Control**: Filters segments based on quality scores
5. **Storage**: Saves to SQLite database with full metadata

### **Output Structure**
```
output/
├── dataset.db              # SQLite database
├── source_files/           # Original files
├── audio_segments/         # Processed segments
├── metadata/               # Legacy metadata
├── temp/                  # Temporary files
└── speaker_alignments/     # Speaker alignment files
```

---

## 🎤 **Step 2: Speaker Alignment**

### **The Problem**
Diarization models assign arbitrary labels (A, B, C) that don't correspond to actual speakers:
- **File 1**: Speaker A = John, Speaker B = Mary
- **File 2**: Speaker A = Mary, Speaker B = John

### **Solution: Web UI (Recommended)**
```bash
# Start the web annotation interface
python dataset_manager.py ui
```

**Access the UI**: `http://localhost:5000`

#### **Web UI Features**
- **Dashboard**: Overview of all files and statistics
- **Audio Playback**: Listen to segments directly in browser
- **Visual Interface**: Drag-and-drop speaker alignment
- **Real-time Preview**: See alignment changes immediately
- **Production-grade**: Based on industry tools like Label Studio

#### **Web UI Workflow**
1. **Dashboard**: Review all processed files
2. **File Selection**: Click on a file to annotate
3. **Audio Review**: Listen to segments for each speaker
4. **Speaker Mapping**: Map speakers to consistent IDs
5. **Save Alignment**: Save alignment configuration
6. **Apply Changes**: Apply alignment to dataset

### **Alternative: Command Line**
```bash
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

---

## 📊 **Step 3: Dataset Management**

### **Check Dataset Status**
```bash
# Get comprehensive dataset status
python dataset_manager.py status

# List all source files
python dataset_manager.py list-files

# List audio segments
python dataset_manager.py list-segments

# List speakers with statistics
python dataset_manager.py list-speakers
```

### **Dataset Statistics**
```bash
# Get detailed statistics
python dataset_manager.py status
```

**Example Output**:
```json
{
  "dataset_directory": "./output",
  "statistics": {
    "source_files": {
      "total": 5,
      "completed": 5,
      "failed": 0,
      "total_size_bytes": 52428800
    },
    "audio_segments": {
      "total": 150,
      "unique_speakers": 8,
      "total_duration_seconds": 1800.5,
      "average_quality": 0.85,
      "average_confidence": 0.92
    }
  }
}
```

### **Quality Control**
```bash
# Validate dataset integrity
python dataset_manager.py validate

# Clean up failed files
python dataset_manager.py cleanup
```

---

## 📤 **Step 4: Export Dataset**

### **Export to Hugging Face Format**
```bash
# Export dataset
python dataset_manager.py export my_dataset --format huggingface
```

### **Export to Orpheus Format**
```bash
# Export for Orpheus TTS training
python dataset_manager.py export my_dataset --format orpheus
```

### **Export Structure**
```
exports/
└── my_dataset/
    ├── audio/
    │   ├── segment_000000.wav
    │   ├── segment_000001.wav
    │   └── ...
    ├── metadata.json
    └── README.md
```

---

## 🚀 **Step 5: Upload to Hugging Face**

### **Install Hugging Face Hub**
```bash
pip install huggingface_hub
```

### **Create Hugging Face Dataset**
```python
from huggingface_utils import prepare_and_upload

# Upload dataset to Hugging Face
prepare_and_upload(
    source_dir="./output",
    repo_name="your-username/your-tts-dataset",
    token="your_huggingface_token",
    dataset_name="My TTS Dataset",
    description="Audio segments with transcriptions for TTS training"
)
```

### **Manual Upload**
```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login

# Create repository
huggingface-cli repo create your-username/your-tts-dataset --type dataset

# Upload files
huggingface-cli upload your-username/your-tts-dataset ./exports/my_dataset/* .
```

---

## 🔧 **Step 6: Production Deployment**

### **Environment Setup**
```bash
# Production environment variables
export ASSEMBLYAI_API_KEY="your_production_key"
export OUTPUT_DIR="/data/tts_datasets"
export BATCH_SIZE=5
export SPEAKER_MAPPING_MODE="interactive"
export PRESERVE_TEMP_FILES=false
```

### **Deployment Options**

#### **Option 1: Direct Python Deployment**
```bash
# Set production environment
export ASSEMBLYAI_API_KEY="your_production_key"
export OUTPUT_DIR="/data/tts_datasets"

# Start web UI
python dataset_manager.py ui --host 0.0.0.0 --port 5000
```

#### **Option 2: Systemd Service (Linux)**
```bash
# Create systemd service file
sudo nano /etc/systemd/system/tts-dataset-maker.service
```

```ini
[Unit]
Description=TTS Dataset Maker
After=network.target

[Service]
Type=simple
User=tts
WorkingDirectory=/opt/tts-dataset-maker
ExecStart=/usr/bin/python3 dataset_manager.py ui --host 0.0.0.0 --port 5000
Restart=always
Environment=ASSEMBLYAI_API_KEY=your_key_here
Environment=OUTPUT_DIR=/data/tts_datasets

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable tts-dataset-maker
sudo systemctl start tts-dataset-maker
```

#### **Option 3: Background Process**
```bash
# Run in background with nohup
nohup python dataset_manager.py ui --host 0.0.0.0 --port 5000 > tts.log 2>&1 &

# Check if running
ps aux | grep dataset_manager

# Stop process
pkill -f dataset_manager
```

---

## 📈 **Step 7: Monitoring & Maintenance**

### **Dataset Monitoring**
```bash
# Check processing status
python dataset_manager.py status

# Monitor failed files
python dataset_manager.py list-files --status failed

# Validate dataset integrity
python dataset_manager.py validate
```

### **Performance Optimization**
```bash
# Process files in batches
python tts_service.py *.mp3 --batch-size 5

# Use parallel processing (future feature)
python tts_service.py *.mp3 --parallel-processing
```

### **Backup & Recovery**
```bash
# Backup database
cp output/dataset.db backup/dataset_$(date +%Y%m%d).db

# Backup audio files
tar -czf backup/audio_segments_$(date +%Y%m%d).tar.gz output/audio_segments/
```

---

## 🎯 **Complete Example Workflow**

### **1. Process Audio Files**
```bash
# Process multiple files
python tts_dataset_maker.py \
    interview1.mp3 \
    interview2.mp3 \
    podcast1.mp3 \
    --output ./my_dataset \
    --name "interview_dataset"
```

### **2. Align Speakers**
```bash
# Start web UI
python dataset_manager.py ui

# Access: http://localhost:5000
# 1. Review files in dashboard
# 2. Click on each file to annotate
# 3. Listen to segments and map speakers
# 4. Save alignment configuration
```

### **3. Export Dataset**
```bash
# Export to Hugging Face format
python dataset_manager.py export interview_dataset --format huggingface
```

### **4. Upload to Hugging Face**
```python
from huggingface_utils import prepare_and_upload

prepare_and_upload(
    source_dir="./output",
    repo_name="myusername/interview-dataset",
    token="hf_your_token_here",
    dataset_name="Interview TTS Dataset",
    description="High-quality interview audio with speaker diarization for TTS training"
)
```

### **5. Verify Upload**
```bash
# Check Hugging Face dataset
# Visit: https://huggingface.co/datasets/myusername/interview-dataset
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **API Key Issues**
```bash
# Check API key
echo $ASSEMBLYAI_API_KEY

# Set API key
export ASSEMBLYAI_API_KEY="your_key_here"
```

#### **Memory Issues**
```bash
# Process files in smaller batches
python tts_service.py *.mp3 --batch-size 1

# Monitor memory usage
htop
```

#### **Storage Issues**
```bash
# Check disk space
df -h

# Clean up temporary files
python dataset_manager.py cleanup
```

#### **Audio Quality Issues**
```bash
# Enable VAD for noisy audio
python tts_service.py *.mp3 --vad

# Adjust quality thresholds in config.py
```

### **Debug Mode**
```bash
# Enable verbose logging
python tts_service.py *.mp3 --verbose

# Debug web UI
python dataset_manager.py ui --debug
```

---

## 📚 **Best Practices**

### **Data Preparation**
1. **Audio Quality**: Use high-quality audio (16kHz+, clear speech)
2. **File Organization**: Organize files by source/type
3. **Naming Convention**: Use descriptive filenames
4. **Backup**: Always backup original files

### **Processing**
1. **Batch Size**: Process files in manageable batches
2. **Quality Control**: Review segments before alignment
3. **Speaker Alignment**: Use web UI for best results
4. **Validation**: Always validate dataset integrity

### **Deployment**
1. **Environment**: Use production environment variables
2. **Monitoring**: Set up logging and monitoring
3. **Backup**: Regular database and file backups
4. **Security**: Secure API keys and file access

### **Hugging Face**
1. **Documentation**: Write comprehensive dataset cards
2. **Licensing**: Choose appropriate licenses
3. **Quality**: Ensure high-quality annotations
4. **Updates**: Regular dataset updates and improvements

---

## 🎉 **Success Metrics**

### **Dataset Quality**
- **Audio Quality**: >80% segments pass quality filters
- **Speaker Alignment**: 100% speakers correctly aligned
- **Transcription Accuracy**: >90% confidence scores
- **Coverage**: All speakers have sufficient data

### **Production Metrics**
- **Processing Time**: <5 minutes per hour of audio
- **Success Rate**: >95% files processed successfully
- **Storage Efficiency**: Optimized file organization
- **Export Time**: <2 minutes for dataset export

---

## 🔗 **Useful Commands Reference**

### **Processing**
```bash
# Simple processing
python tts_dataset_maker.py *.mp3

# Advanced processing
python tts_service.py *.mp3 --output-dir ./output --verbose

# Batch processing
python tts_service.py *.mp3 --batch-size 5
```

### **Management**
```bash
# Check status
python dataset_manager.py status

# List files
python dataset_manager.py list-files

# List segments
python dataset_manager.py list-segments

# Validate dataset
python dataset_manager.py validate
```

### **Speaker Alignment**
```bash
# Web UI
python dataset_manager.py ui

# Command line
python dataset_manager.py create-reference
python dataset_manager.py create-workflow
python dataset_manager.py apply-alignment alignment.json
```

### **Export**
```bash
# Export dataset
python dataset_manager.py export my_dataset --format huggingface

# Upload to Hugging Face
python -c "from huggingface_utils import prepare_and_upload; prepare_and_upload(...)"
```

---

## 🎯 **Conclusion**

This complete workflow takes you from raw audio files to a production-ready Hugging Face dataset with:

- ✅ **Multi-file processing** with intelligent storage
- ✅ **Speaker alignment** with web UI
- ✅ **Quality control** and validation
- ✅ **Export capabilities** for multiple formats
- ✅ **Hugging Face integration** for dataset publishing
- ✅ **Production deployment** with monitoring

The system is designed for **production use** with enterprise-grade features, comprehensive error handling, and scalable architecture.

**Ready to create your TTS dataset!** 🚀
