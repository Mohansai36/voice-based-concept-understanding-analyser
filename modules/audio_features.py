"""
Audio Feature Extraction & Scoring Engine
Epic 2: Core Logic Development (Speech Understanding & Evaluation Engine)

Implements the fluency-analysis requirements exactly as specified:
- Filler word usage (um, like, uh, etc.)
- Pause ratio (proportion of silence vs. total duration)
- RMS energy levels (via Librosa) as a proxy for speaking confidence/clarity
"""

import os
import re
import tempfile
import numpy as np
import librosa

FILLER_WORDS = ["um", "umm", "uh", "uhh", "like", "you know", "actually", "basically", "i mean", "so"]


def _load_audio(audio_bytes: bytes, original_format: str = "wav"):
    """Loads audio into a numpy waveform array + sample rate via Librosa (handles wav/mp3/m4a/ogg via ffmpeg)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{original_format}") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        y, sr_rate = librosa.load(tmp_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=y, sr=sr_rate)
        return y, sr_rate, duration
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def detect_filler_words(transcript: str) -> dict:
    """Counts filler word occurrences in the transcript."""
    text_lower = transcript.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    total_words = len(words) if words else 1

    breakdown = {}
    filler_count = 0
    for filler in FILLER_WORDS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        count = len(re.findall(pattern, text_lower))
        if count > 0:
            breakdown[filler] = count
            filler_count += count

    filler_ratio = round((filler_count / total_words) * 100, 1)
    return {"filler_count": filler_count, "filler_ratio_pct": filler_ratio, "breakdown": breakdown, "total_words": total_words}


def analyze_audio(audio_bytes: bytes, transcript: str, original_format: str = "wav") -> dict:
    """
    Returns:
        dict with keys:
            duration_sec, pause_ratio_pct, pause_count,
            rms_energy_mean, rms_energy_std,
            filler_count, filler_ratio_pct, filler_breakdown,
            fluency_score (0-100), fluency_verdict,
            waveform (y array), sample_rate
    """
    try:
        y, sr_rate, duration = _load_audio(audio_bytes, original_format)
    except Exception as e:
        return {"error": f"Audio analysis failed: {e}"}

    if duration <= 0:
        return {"error": "Audio duration is zero."}

    # --- Pause ratio: proportion of the clip that is silence ---
    non_silent_intervals = librosa.effects.split(y, top_db=30)
    voiced_duration = sum((end - start) for start, end in non_silent_intervals) / sr_rate
    pause_duration = max(0.0, duration - voiced_duration)
    pause_ratio_pct = round((pause_duration / duration) * 100, 1)
    pause_count = max(0, len(non_silent_intervals) - 1)

    # --- RMS energy: speaking clarity/confidence proxy ---
    rms = librosa.feature.rms(y=y)[0]
    rms_energy_mean = float(np.mean(rms))
    rms_energy_std = float(np.std(rms))

    # --- Filler words (from transcript) ---
    filler_info = detect_filler_words(transcript)

    # --- Combine into a single fluency score (0-100) ---
    pause_score = max(0, 100 - pause_ratio_pct * 1.5)
    filler_score = max(0, 100 - filler_info["filler_ratio_pct"] * 4)
    # RMS values are typically small floats (~0.01-0.1); scale to a 0-100 range
    energy_score = min(100, rms_energy_mean * 900)

    fluency_score = round((pause_score * 0.4) + (filler_score * 0.35) + (energy_score * 0.25), 1)

    if fluency_score >= 80:
        verdict = "Confident and fluent delivery"
    elif fluency_score >= 60:
        verdict = "Reasonably fluent, some hesitation"
    elif fluency_score >= 40:
        verdict = "Noticeable hesitation or filler word usage"
    else:
        verdict = "Delivery needs significant improvement"

    return {
        "duration_sec": round(duration, 1),
        "pause_ratio_pct": pause_ratio_pct,
        "pause_count": pause_count,
        "rms_energy_mean": round(rms_energy_mean, 4),
        "rms_energy_std": round(rms_energy_std, 4),
        "filler_count": filler_info["filler_count"],
        "filler_ratio_pct": filler_info["filler_ratio_pct"],
        "filler_breakdown": filler_info["breakdown"],
        "fluency_score": fluency_score,
        "fluency_verdict": verdict,
        "waveform": y,
        "sample_rate": sr_rate,
    }
