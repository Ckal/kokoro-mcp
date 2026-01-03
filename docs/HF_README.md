---
title: Kokoro Text-to-Audio
emoji: 🎵
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.28.0
app_file: app.py
pinned: false
---

# Kokoro Text-to-Audio App

A simple Gradio application that uses the hexgrad/Kokoro-82M model to convert text to audio.

## Setup Instructions

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python kokoro_text_to_audio.py
   ```

3. Open your web browser and navigate to the URL displayed in the terminal (typically http://127.0.0.1:7860)

## Features

- Simple text input box for entering the text you want to convert to audio
- Adjustable speech speed slider
- Audio playback directly in the browser

## Requirements

- Python 3.8 or higher
- GPU is recommended for faster generation, but not required
- Internet connection (to download the model on first run)

## Model Information

This app uses the [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model from Hugging Face.