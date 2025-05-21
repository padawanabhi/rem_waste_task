# English Accent Detection Tool

This tool accepts a public video URL (YouTube, Loom, direct MP4, etc.), extracts the audio, transcribes it, detects the language, and classifies the English accent (e.g., American, British, Australian, etc.) using an open-source model. It also provides a transcription display and user feedback option.

## Features
- Accepts public video URLs
- Extracts audio using ffmpeg/yt-dlp
- Transcribes using Whisper
- Detects spoken language (supports only English for accent detection)
- Classifies English accent (using HuggingFace SpeechBrain model)
- Outputs accent, confidence score, top-3 probabilities, and a probability bar chart
- Displays transcription in an expandable section
- User feedback option
- Improved logging and error handling

## App Workflow
1. **Input**: User enters a public video URL (YouTube, Loom, direct MP4, etc.).
2. **Audio Extraction**: The app downloads the video and extracts the audio as a WAV file.
3. **Transcription & Language Detection**: The app transcribes the audio and detects the spoken language using OpenAI Whisper.
4. **Accent Detection**: If the language is English, the app classifies the accent and displays the predicted accent, confidence, top-3 probabilities, and a probability bar chart.
5. **Transcription Display**: Users can view the full transcription in an expandable section.
6. **Feedback**: Users can provide feedback on the prediction.

## Setup

1. **Clone the repository**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Install ffmpeg and yt-dlp**

- On macOS (with Homebrew):
  ```bash
  brew install ffmpeg
  ```
- On Ubuntu:
  ```bash
  sudo apt-get install ffmpeg
  ```

4. **(For Streamlit Cloud deployment)**
   - Add a `packages.txt` file in the repo root with:
     ```
     ffmpeg
     ```

5. **Run the app locally**

```bash
streamlit run app.py
```

6. **Usage**
- Enter a public video URL in the input box.
- Click "Analyze Accent".
- Wait for the audio to be processed and the accent to be detected.
- View the predicted accent, confidence, top-3 probabilities, and the full probability bar chart.
- Optionally, expand to view the transcription and provide feedback on the prediction.

## Understanding Confidence and Probabilities
- The model outputs a probability for each accent. The highest probability is shown as the "confidence" for the predicted accent.
- **Low confidence (e.g., under 20%)** means the model is unsure and sees several possible accents. This is common for short, unclear, or ambiguous audio, or for accents not well represented in the model.
- **High confidence (e.g., over 80%)** means the model is very sure about one accent. However, high confidence does not guarantee correctness, especially for regional or rare accents.
- The bar chart shows the model's probability for each accent class.

## Disclaimer
This tool uses an open-source accent detection model. Results may not be accurate for all English accents, especially regional or less common ones. Please use for demonstration or research purposes only, not for high-stakes decisions.
