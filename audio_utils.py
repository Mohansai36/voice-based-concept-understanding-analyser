"""
audio_utils.py
--------------
Audio loading, feature extraction, and waveform visualization utilities
for the Voice-Based Concept Understanding Analyser (VBCUA).

Covers Epic 2 / Story: "Audio Feature Extraction & Scoring Engine"
- pause_ratio
- rms_energy
- zero_crossing_rate
- duration_sec
- waveform plot generation (for UI + PDF report)
"""

import os
import io
import uuid
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for Streamlit/server use
import matplotlib.pyplot as plt


SUPPORTED_FORMATS = (".wav", ".mp3")


def load_audio(file_path: str, target_sr: int = 16000):
    """
    Load an audio file and resample to a consistent sample rate.

    Args:
        file_path: path to the audio file (WAV/MP3)
        target_sr: target sample rate for consistent processing

    Returns:
        (y, sr): waveform samples (np.ndarray) and sample rate
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format '{ext}'. Supported: {SUPPORTED_FORMATS}")

    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return y, sr


def save_uploaded_file(uploaded_file, upload_dir: str = "uploads") -> str:
    """
    Persist a Streamlit UploadedFile object to disk and return its path.
    """
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(uploaded_file.name)[1].lower() or ".wav"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(upload_dir, unique_name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def compute_pause_ratio(y: np.ndarray, sr: int, top_db: int = 30) -> float:
    """
    Estimate the ratio of silence/pause duration to total duration using
    librosa's non-silent interval detection.
    """
    if len(y) == 0:
        return 1.0

    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    voiced_samples = sum((end - start) for start, end in non_silent_intervals)
    total_samples = len(y)

    if total_samples == 0:
        return 1.0

    pause_ratio = 1.0 - (voiced_samples / total_samples)
    return float(np.clip(pause_ratio, 0.0, 1.0))


def compute_rms_energy(y: np.ndarray) -> float:
    """
    Compute the mean RMS (root-mean-square) energy — a proxy for vocal
    confidence / loudness consistency.
    """
    if len(y) == 0:
        return 0.0
    rms = librosa.feature.rms(y=y)[0]
    return float(np.mean(rms))


def compute_zero_crossing_rate(y: np.ndarray) -> float:
    """
    Compute the mean zero-crossing rate — useful signal for speech
    clarity / noisiness.
    """
    if len(y) == 0:
        return 0.0
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return float(np.mean(zcr))


def extract_audio_features(file_path: str) -> dict:
    """
    Extract the full AUDIO_FEATURE record for a given audio file, matching
    the ERD's AUDIO_FEATURE entity:
        pause_ratio, rms_energy, zero_crossing_rate, duration_sec
    """
    y, sr = load_audio(file_path)
    duration_sec = float(librosa.get_duration(y=y, sr=sr))

    features = {
        "pause_ratio": compute_pause_ratio(y, sr),
        "rms_energy": compute_rms_energy(y),
        "zero_crossing_rate": compute_zero_crossing_rate(y),
        "duration_sec": duration_sec,
    }
    return features


def save_waveform(file_path: str, output_dir: str = "waveforms") -> str:
    """
    Render and save a waveform plot image (PNG) for a given audio file.
    Used for both the Streamlit UI and the PDF report.

    Returns:
        Path to the saved waveform PNG image.
    """
    os.makedirs(output_dir, exist_ok=True)
    y, sr = load_audio(file_path)

    fig, ax = plt.subplots(figsize=(8, 3))
    times = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(times, y, linewidth=0.6, color="#3b82f6")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title("Audio Waveform")
    fig.tight_layout()

    out_name = f"{uuid.uuid4().hex}_waveform.png"
    out_path = os.path.join(output_dir, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path
