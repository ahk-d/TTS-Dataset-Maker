#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tts-dataset-maker",
    version="1.0.0",
    author="TTS Dataset Maker",
    description="A modular tool for creating and exploring TTS datasets from YouTube videos using AssemblyAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tts-dataset-maker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "tts-dataset-maker=main:main",
        ],
    },
    keywords="tts, text-to-speech, dataset, youtube, assemblyai, audio, transcription",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/tts-dataset-maker/issues",
        "Source": "https://github.com/yourusername/tts-dataset-maker",
    },
) 