#!/usr/bin/env python3.10

# Set environment variable to suppress Streamlit watcher errors
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Import and configure warnings and logging
import warnings
import logging
from typing import Tuple

# Configure logging to show time, level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
# Redirect warnings to logging
def custom_warning(message, category, filename, lineno, file=None, line=None):
    logging.warning(f"{category.__name__}: {message} ({filename}:{lineno})")
warnings.showwarning = custom_warning
# Suppress specific and general warnings
warnings.filterwarnings("ignore", message="CategoricalEncoder.expect_len was never called*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Standard libraries
import tempfile
import os
import subprocess
import requests
from pathlib import Path
import math
import shutil

# Third-party libraries
import streamlit as st
import torch
import whisper
import librosa
import numpy as np
import soundfile as sf
from speechbrain.pretrained import EncoderClassifier
import pandas as pd

# -----------------------------
# Utility Functions
# -----------------------------

def download_and_extract_audio(url: str, output_audio_path: str) -> str:
    """
    Downloads a video from a public URL using yt-dlp and extracts the audio as a WAV file.
    The audio is saved to output_audio_path.
    Returns the path to the extracted WAV file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        command = [
            "yt-dlp",
            "-x", "--audio-format", "wav",
            "-o", os.path.join(tmpdir, "%(title)s.%(ext)s"),
            url
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"yt-dlp stdout: {result.stdout.decode()}")
        logging.info(f"yt-dlp stderr: {result.stderr.decode()}")
        if result.returncode != 0:
            logging.error(f"yt-dlp failed: {result.stderr.decode()}")
            raise RuntimeError(f"yt-dlp failed: {result.stderr.decode()}")
        logging.info(f"Files in tmpdir: {os.listdir(tmpdir)}")
        for fname in os.listdir(tmpdir):
            if fname.endswith(".wav"):
                shutil.copyfile(os.path.join(tmpdir, fname), output_audio_path)
                return output_audio_path
        logging.error("yt-dlp did not produce a .wav file")
        raise RuntimeError("yt-dlp did not produce a .wav file")

def transcribe_and_check_english(audio_path: str) -> Tuple[str, str]:
    """
    Transcribes the given audio file using OpenAI Whisper and detects the language.
    Returns the transcription text and the detected language code.
    """
    model = whisper.load_model('base')
    result = model.transcribe(audio_path)
    text: str = result['text']
    lang: str = result['language']
    return text, lang

def ensure_wav_16k_mono(input_audio_path: str) -> str:
    """
    Converts any audio file to 16kHz mono WAV using librosa and soundfile.
    Returns the path to the new WAV file.
    """
    y: np.ndarray
    sr: int
    y, sr = librosa.load(input_audio_path, sr=16000, mono=True)
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_wav.name, y, 16000)
    return temp_wav.name

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax for a numpy array.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# -----------------------------
# Streamlit App UI and Logic
# -----------------------------

st.title('English Accent Detection Tool')

# Disclaimer for users
st.markdown('''<span style="color:red"><b>Disclaimer:</b> This tool uses an open-source accent detection model. Results may not be accurate for all English accents, especially regional or less common ones. Please use for demonstration or research purposes only, not for high-stakes decisions.</span>''', unsafe_allow_html=True)

# Input for video URL
video_url: str = st.text_input('Enter a public video URL (YouTube, Loom, direct MP4, etc.):')

feedback: str = None

if st.button('Analyze Accent') and video_url:
    # Step 1: Download and extract audio
    with st.spinner('Processing video and extracting audio...'):
        try:
            audio_path: str = download_and_extract_audio(video_url, "audio.wav")
            wav_16k_path: str = ensure_wav_16k_mono(audio_path)
        except Exception as e:
            logging.error(f"Error extracting audio: {e}")
            st.error(f"Error extracting audio: {e}")
            st.stop()

    st.success("Audio extracted. Running transcription and accent detection...")

    # Step 2: Transcribe and check language
    with st.spinner('Transcribing and detecting language...'):
        try:
            transcription: str
            lang: str
            transcription, lang = transcribe_and_check_english(wav_16k_path)
            is_english: bool = (lang == 'en')
            lang_status: str = "English" if is_english else f"Not English (detected: {lang})"
            st.markdown(f"**Detected Language:** {lang_status}")
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            st.error(f"Error during transcription: {e}")
            st.stop()

    # Step 3: Accent detection (only if English)
    if is_english:
        with st.spinner('Detecting accent...'):
            try:
                classifier: EncoderClassifier = EncoderClassifier.from_hparams(source="Jzuluaga/accent-id-commonaccent_ecapa")
                prediction = classifier.classify_file(wav_16k_path)
                accent_labels: list[str] = list(classifier.hparams.label_encoder.lab2ind.keys())
                probs: np.ndarray = prediction[0].detach().cpu().numpy().flatten()
                probs_np: np.ndarray = np.array(probs, dtype=np.float32, copy=True)
                if probs_np.ndim == 1 and len(probs_np) > 0 and np.sum(probs_np) > 0:
                    probs_softmax: np.ndarray = softmax(probs_np)
                    # Get top 3 accents
                    top_n: int = 3
                    top_indices: np.ndarray = probs_softmax.argsort()[-top_n:][::-1]
                    top_accents: list[Tuple[str, float]] = [
                        (accent_labels[idx], probs_softmax[idx]*100) for idx in top_indices if probs_softmax[idx] > 0
                    ]
                    st.markdown(f"### Predicted Accent: **{top_accents[0][0]}**")
                    st.markdown(f"**Confidence:** {top_accents[0][1]:.1f}%")
                    if len(top_accents) > 1:
                        st.markdown("#### Top 3 Accent Probabilities:")
                        st.table(pd.DataFrame(top_accents, columns=["Accent", "Probability (%)"]))
                    chart_data: pd.DataFrame = pd.DataFrame({
                        'Accent': accent_labels,
                        'Probability (%)': [p * 100 for p in probs_softmax]
                    })
                    st.markdown("#### All Accent Probabilities:")
                    st.bar_chart(
                        chart_data.set_index('Accent'),
                        use_container_width=True
                    )
                    st.info("**Note:** Low confidence (e.g., under 20%) means the model is unsure and sees several possible accents. This is common for short, unclear, or ambiguous audio, or for accents not well represented in the model. High confidence means the model is very sure about one accent, but does not guarantee correctness.")
                else:
                    st.warning("Could not detect accent from the audio.")
            except Exception as ex:
                logging.error(f"Error in probability processing: {ex}")
                st.warning("Could not detect accent from the audio.")
    else:
        st.warning("Accent detection is only supported for English language audio.")

    # Step 4: Transcription display (expander)
    with st.expander("Show Transcription"):
        st.markdown("**Transcription:**")
        st.write(transcription)

    # Step 5: Clean up temp audio file
    os.unlink(audio_path)

    # Step 6: Feedback option
    st.markdown("---")
    st.markdown("### Was this prediction correct?")
    feedback = st.radio("Feedback", ("Yes", "No"), horizontal=True)
    if st.button("Submit Feedback"):
        if feedback == "Yes":
            st.success("Thank you for your feedback!")
        elif feedback == "No":
            st.warning("Thank you for your feedback! We'll use this to improve the tool.") 