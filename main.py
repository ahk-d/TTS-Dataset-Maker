#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Dataset Maker - Main Entry Point

This script launches the Gradio interface for exploring TTS dataset segments.
It requires the output files from the TTS dataset maker to be present.
"""

import os
import sys
from data_processor import DataProcessor
from ui_components import UIComponents

def main():
    """Main function to launch the TTS Dataset Explorer."""
    print("TTS Dataset Maker - Explorer Interface")
    print("=" * 50)
    
    # Import gradio
    import gradio as gr
    
    # Initialize data processor
    try:
        data_processor = DataProcessor()
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure you have run the TTS dataset maker first and have the output files.")
        return
    
    # Create and launch UI
    print("Creating interface...")
    ui = UIComponents(data_processor)
    demo = ui.create_interface()
    
    print("Launching Gradio interface...")
    demo.launch()

if __name__ == "__main__":
    main() 