import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import tempfile
import os
import subprocess
import requests
from pathlib import Path
import torch
import whisper
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import librosa
import numpy as np
import soundfile as sf
from speechbrain.pretrained import EncoderClassifier
import math
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Download video and extract audio
def download_and_extract_audio(url, output_audio_path):
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
                os.rename(os.path.join(tmpdir, fname), output_audio_path)
                return output_audio_path
        logging.error("yt-dlp did not produce a .wav file")
        raise RuntimeError("yt-dlp did not produce a .wav file")

# Transcribe and check language
def transcribe_and_check_english(audio_path):
    model = whisper.load_model('base')
    result = model.transcribe(audio_path)
    text = result['text']
    lang = result['language']
    return text, lang

# Accent classification (using a pre-trained model)
def classify_accent(audio_path):
    # For demo, use a simple classifier (replace with a real accent classifier if available)
    # Here, we use a placeholder that always returns 'American' with 80% confidence
    # You can replace this with a HuggingFace model if available
    return 'American', 0.8, 'Demo: Replace with a real accent classifier for production.'

def ensure_wav_16k_mono(input_audio_path):
    # Convert audio to 16kHz mono WAV using librosa and soundfile
    y, sr = librosa.load(input_audio_path, sr=16000, mono=True)
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_wav.name, y, 16000)
    return temp_wav.name

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

st.title('English Accent Detection Tool')

st.markdown('''<span style="color:red"><b>Disclaimer:</b> This tool uses an open-source accent detection model. Results may not be accurate for all English accents, especially regional or less common ones. Please use for demonstration or research purposes only, not for high-stakes decisions.</span>''', unsafe_allow_html=True)

video_url = st.text_input('Enter a public video URL (YouTube, Loom, direct MP4, etc.):')

feedback = None

if st.button('Analyze Accent') and video_url:
    with st.spinner('Processing video and extracting audio...'):
        try:
            audio_path = download_and_extract_audio(video_url, "audio.wav")
            wav_16k_path = ensure_wav_16k_mono(audio_path)
        except Exception as e:
            logging.error(f"Error extracting audio: {e}")
            st.error(f"Error extracting audio: {e}")
            st.stop()

    st.success("Audio extracted. Running accent detection...")
    with st.spinner('Detecting accent...'):
        try:
            classifier = EncoderClassifier.from_hparams(source="Jzuluaga/accent-id-commonaccent_ecapa")
            prediction = classifier.classify_file(wav_16k_path)
            label = prediction[3][0]
            probs = prediction[0].detach().cpu().numpy().flatten()
            accent_labels = classifier.hparams.label_encoder.lab2ind.keys()
            accent_labels = list(accent_labels)
            # Normalize probabilities with softmax if needed
            try:
                probs_np = np.array(probs, dtype=np.float32, copy=True)
                if probs_np.ndim == 1 and len(probs_np) > 0 and np.sum(probs_np) > 0:
                    probs_softmax = softmax(probs_np)
                    # Get top 3 accents
                    top_n = 3
                    top_indices = probs_softmax.argsort()[-top_n:][::-1]
                    top_accents = [(accent_labels[idx], probs_softmax[idx]*100) for idx in top_indices if probs_softmax[idx] > 0]
                    # Show only one predicted accent and its confidence
                    st.markdown(f"### Predicted Accent: **{top_accents[0][0]}**")
                    st.markdown(f"**Confidence:** {top_accents[0][1]:.1f}%")
                    # Table of top 3 (if at least 2 nonzero)
                    if len(top_accents) > 1:
                        st.markdown("#### Top 3 Accent Probabilities:")
                        st.table(pd.DataFrame(top_accents, columns=["Accent", "Probability (%)"]))
                    # Bar chart with labels
                    chart_data = pd.DataFrame({
                        'Accent': accent_labels,
                        'Probability (%)': [p * 100 for p in probs_softmax]
                    })
                    st.markdown("#### All Accent Probabilities:")
                    st.bar_chart(
                        chart_data.set_index('Accent'),
                        use_container_width=True
                    )
                    # Add user-facing message about confidence
                    st.info("**Note:** Low confidence (e.g., under 20%) means the model is unsure and sees several possible accents. This is common for short, unclear, or ambiguous audio, or for accents not well represented in the model. High confidence means the model is very sure about one accent, but does not guarantee correctness.")
                else:
                    st.warning("Could not detect accent from the audio.")
            except Exception as ex:
                logging.error(f"Error in probability processing: {ex}")
                st.warning("Could not detect accent from the audio.")
        except Exception as e:
            logging.error(f"Error during accent detection: {e}")
            st.error(f"Error during accent detection: {e}")
            st.stop()
    os.unlink(audio_path)

    # Feedback option
    st.markdown("---")
    st.markdown("### Was this prediction correct?")
    feedback = st.radio("Feedback", ("Yes", "No"), horizontal=True)
    if st.button("Submit Feedback"):
        if feedback == "Yes":
            st.success("Thank you for your feedback!")
        elif feedback == "No":
            st.warning("Thank you for your feedback! We'll use this to improve the tool.") 