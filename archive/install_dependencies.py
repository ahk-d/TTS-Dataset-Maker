#!/usr/bin/env python3
"""
Installation script for TTS Dataset Maker dependencies
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing TTS Dataset Maker dependencies...")
    
    try:
        # Install pydantic-settings first to avoid import issues
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic-settings>=2.0.0"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_assemblyai_key():
    """Check if AssemblyAI API key is set"""
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("⚠️  AssemblyAI API key not found!")
        print("Please set your API key:")
        print("export ASSEMBLYAI_API_KEY='your_api_key_here'")
        print("Or create a .env file with:")
        print("ASSEMBLYAI_API_KEY=your_api_key_here")
        return False
    else:
        print("✅ AssemblyAI API key found!")
        return True

def create_env_file():
    """Create example .env file"""
    env_content = """# AssemblyAI API Configuration
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here

# Processing Configuration
DEFAULT_SAMPLE_RATE=24000
MIN_SEGMENT_DURATION=1.0
MAX_SEGMENT_DURATION=30.0
MIN_CONFIDENCE_SCORE=0.7
AUDIO_OVERLAP_MS=100

# Output Configuration
OUTPUT_DIR=./output
TEMP_DIR=./temp
"""
    
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env file template")
        print("Please edit .env file and add your AssemblyAI API key")
    else:
        print("ℹ️  .env file already exists")

def main():
    """Main installation function"""
    print("🚀 TTS Dataset Maker Setup")
    print("=" * 40)
    
    # Install dependencies
    if not install_requirements():
        return 1
    
    # Create .env file
    create_env_file()
    
    # Check API key
    check_assemblyai_key()
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Add your AssemblyAI API key to .env file")
    print("2. Run: python tts_service.py --help")
    print("3. Process your audio files!")
    
    return 0

if __name__ == "__main__":
    exit(main())
