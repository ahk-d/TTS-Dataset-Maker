#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized Dependency Installer for TTS Dataset Maker

This script handles all package installation in one place, ensuring
consistent and clean dependency management.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package, upgrade=False):
    """Install a single package with error handling."""
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ {package}")
            return True
        else:
            print(f"✗ {package}: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        print(f"✗ {package}: {e}")
        return False

def install_requirements():
    """Install all required packages from requirements.txt."""
    print("📦 Installing dependencies from requirements.txt...")
    
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All dependencies installed successfully!")
            return True
        else:
            print(f"❌ Failed to install requirements: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def install_optional_dependencies():
    """Install optional dependencies for advanced features."""
    print("\n🔧 Installing optional dependencies...")
    
    optional_packages = [
        "scipy",  # For advanced denoising
        "torch",  # For DeepFilterNet
    ]
    
    success_count = 0
    for package in optional_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Optional dependencies: {success_count}/{len(optional_packages)} installed")
    return success_count == len(optional_packages)

def check_installation():
    """Check if all required packages are installed."""
    print("\n🔍 Checking installation...")
    
    required_packages = [
        "yt_dlp",
        "assemblyai", 
        "gradio",
        "pydub",
        "pandas",
        "numpy",
        "soundfile"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True

def main():
    """Main installation function."""
    print("🎵 TTS Dataset Maker - Dependency Installer")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found. Please run this from the TTS-Dataset-Maker directory.")
        return 1
    
    # Install core dependencies
    if not install_requirements():
        print("❌ Failed to install core dependencies.")
        return 1
    
    # Install optional dependencies
    install_optional_dependencies()
    
    # Verify installation
    if not check_installation():
        print("❌ Installation verification failed.")
        return 1
    
    print("\n🎉 Installation completed successfully!")
    print("\n🚀 You can now run:")
    print("   python tts_dataset_maker.py 'YOUR_URL' --assemblyai-key 'YOUR_KEY'")
    print("   python main.py")
    print("   python metadata_generator.py")
    
    return 0

if __name__ == "__main__":
    exit(main()) 