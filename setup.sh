#!/bin/bash
#chmod +x setup.sh

# Clone repositories
git clone https://github.com/AI4Bharat/IndicTrans2.git
git clone https://github.com/VarunGumma/IndicTransToolkit.git

# Install system dependencies
sudo apt-get update
sudo apt-get install -y ffmpeg  # Required for pydub

# Setup IndicTrans2 environment
cd IndicTrans2/huggingface_interface

# Install Python packages
python3 -m pip install -r ../requirements.txt
python3 -c "import nltk; nltk.download('punkt')"

# Install IndicTransToolkit
cd ../../IndicTransToolkit
python3 -m pip install --editable .
cd ..

# Install remaining requirements
python3 -m pip install yt-dlp openai-whisper TTS pydub

echo "Installation completed successfully!"