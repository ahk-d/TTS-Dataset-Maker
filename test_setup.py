#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Setup Script for TTS Dataset Maker

This script tests the setup and verifies all components work correctly.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import gradio as gr
        print("✓ Gradio imported successfully")
    except ImportError as e:
        print(f"✗ Gradio import failed: {e}")
        return False
    
    try:
        import soundfile as sf
        print("✓ SoundFile imported successfully")
    except ImportError as e:
        print(f"✗ SoundFile import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    return True

def test_modules():
    """Test that our custom modules can be imported."""
    print("\n🔍 Testing custom modules...")
    
    try:
        from data_processor import DataProcessor
        print("✓ DataProcessor imported successfully")
    except ImportError as e:
        print(f"✗ DataProcessor import failed: {e}")
        return False
    
    try:
        from ui_components import UIComponents
        print("✓ UIComponents imported successfully")
    except ImportError as e:
        print(f"✗ UIComponents import failed: {e}")
        return False
    
    try:
        from metadata_generator import MetadataGenerator
        print("✓ MetadataGenerator imported successfully")
    except ImportError as e:
        print(f"✗ MetadataGenerator import failed: {e}")
        return False
    
    return True

def test_files():
    """Test that all required files exist."""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        "main.py",
        "data_processor.py", 
        "ui_components.py",
        "metadata_generator.py",
        "youtube_processor.py",
        "tts_dataset_maker.py",
        "requirements.txt",
        "README.md",
        "LICENSE",
        "setup.py",
        "CONTRIBUTING.md",
        ".gitignore"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    return True

def test_output_directories():
    """Test that output directories can be created."""
    print("\n🔍 Testing output directories...")
    
    try:
        # Test creating output directories
        Path("output").mkdir(exist_ok=True)
        Path("output/segments").mkdir(exist_ok=True)
        Path("output/segments/audio").mkdir(exist_ok=True)
        print("✓ Output directories created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create output directories: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 TTS Dataset Maker - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_modules,
        test_files,
        test_output_directories
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! The setup is ready.")
        print("\n🎯 Next steps:")
        print("   1. Get an AssemblyAI API key")
        print("   2. Run: python tts_dataset_maker.py 'YOUR_YOUTUBE_URL' --assemblyai-key 'YOUR_KEY'")
        print("   3. Run: python main.py (to explore the dataset)")
        print("   4. Run: python metadata_generator.py (to create training data)")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 