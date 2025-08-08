# Contributing to TTS Dataset Maker

Thank you for your interest in contributing to TTS Dataset Maker! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and single-purpose

### Testing
- Test your changes before submitting
- Ensure the interface works correctly
- Test with different audio files and formats

### Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** following the guidelines above
3. **Test thoroughly** with different scenarios
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description

### Commit Messages
Use clear, descriptive commit messages:
- `feat: add new metadata generation feature`
- `fix: resolve audio loading issue`
- `docs: update README with new usage examples`

## Project Structure

```
TTS-Dataset-Maker/
├── main.py                    # Main entry point
├── data_processor.py          # Data handling
├── ui_components.py           # Gradio interface
├── youtube_processor.py       # YouTube processing
├── metadata_generator.py      # Metadata generation
├── requirements.txt           # Dependencies
├── setup.py                  # Package setup
├── README.md                 # Documentation
├── LICENSE                   # MIT License
└── .gitignore               # Git ignore rules
```

## Areas for Contribution

- **Bug fixes**: Report and fix issues
- **Feature enhancements**: Add new functionality
- **Documentation**: Improve README and docstrings
- **Testing**: Add unit tests and integration tests
- **Performance**: Optimize audio processing
- **UI improvements**: Enhance the Gradio interface

## Questions or Issues?

- **Bug reports**: Use GitHub Issues
- **Feature requests**: Use GitHub Issues
- **Questions**: Use GitHub Discussions

Thank you for contributing! 🎵 