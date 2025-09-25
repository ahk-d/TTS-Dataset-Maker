#!/usr/bin/env python3
"""
Simple setup script for TTS Dataset Maker
"""
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    try:
        # Try uv first
        subprocess.run(["uv", "sync"], check=True)
        print("✅ Dependencies installed with uv")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ Dependencies installed with pip")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False

def create_env_file():
    """Create .env file from template"""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file and add your AssemblyAI API key")
        return True
    elif env_file.exists():
        print("✅ .env file already exists")
        return True
    else:
        print("❌ No env.example file found")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["output", "output/exports", "configs"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up TTS Dataset Maker...")
    
    success = True
    success &= install_dependencies()
    success &= create_env_file()
    success &= create_directories()
    
    if success:
        print("\n🎉 Setup complete!")
        print("\nNext steps:")
        print("1. Edit .env file and add your AssemblyAI API key")
        print("2. Create a config file in configs/")
        print("3. Run: python local_processor.py configs/your_config.json")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
