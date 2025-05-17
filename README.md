# 🎵 AudioBridge: Cross-Lingual Audio Translation Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Flask](https://img.shields.io/badge/Flask-2.0+-blue.svg)](https://flask.palletsprojects.com/)

**An end-to-end system that translates audio content from videos into target languages while preserving speech characteristics**

![System Architecture](assets/pipeline.png)

## ✨ Features

- **YouTube Video Processing**  
  Extract audio and captions directly from YouTube URLs
- **AI-Powered Translation**  
  State-of-the-art multilingual translation using IndicTrans2
- **Neural Voice Cloning**  
  Maintain speaker characteristics with XTTS-v2 text-to-speech
- **Web Interface**  
  User-friendly Flask-based UI for easy interaction
- **Batch Processing**  
  Efficient handling of long-form content with smart batching
- **Multi-Format Support**  
  Automatic subtitle conversion (VTT/SRT to plain text)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- FFmpeg (`sudo apt install ffmpeg` on Linux)
- NVIDIA GPU with CUDA (recommended)

### Installation

# Clone repository
```bash
git clone https://github.com/YOUR_USERNAME/audio-translator.git
cd audio-translato

# Install dependencies
pip install -r requirements.txt

# Download NLP resources
python -m nltk.downloader punkt
```

## Usage

```bash
python main.py
```
Visit http://localhost:5000 in your browser.

###🔧 Configuration

Create a .env file:
```bash
MODEL_PRECISION=half  # half|full
TARGET_LANG=hin_Deva  # ISO language code
BATCH_SIZE=4          # Translation batch size
MAX_INPUT_LENGTH=300  # Character limit for safety
```

## 🌍 Supported Languages

Language	Code	TTS Support
Hindi	hin_Deva	✅
Tamil	tam_Taml	✅
Spanish	spa_Latn	⏳
French	fra_Latn	⏳

## 📂 Project Structure
```bash
audio-translator/
├── utils/              # Core processing modules
│   ├── downloader.py
│   ├── translator.py
│   ├── sentence_splitter.py
│   └── tts_generator.py
├── templates/          # Flask HTML templates
│   └── index.html
├── uploads/            # User-generated content
├── main.py             # Flask application entry
└── requirements.txt    # Dependency list
```
## 📚 Model Architecture
Audio Extraction: YouTube DL + Whisper ASR

Text Processing: NLTK sentence splitting

Translation: IndicTrans2 (1B parameter model)

Speech Synthesis: XTTS-v2 with voice cloning

## 🤝 Contributing
Contributions are welcome! Please open an issue first to discuss proposed changes.

## 🙏 Acknowledgements
OpenAI Whisper for speech recognition.

AI4Bharat for IndicTrans2 models.

Coqui TTS for XTTS-v2 implementation.
