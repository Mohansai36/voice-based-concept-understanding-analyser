"""
Speech-to-Text Module
Epic 2: Core Logic Development (Speech Understanding & Evaluation Engine)

Uses OpenAI Whisper (open-source, runs locally) to transcribe spoken audio
into text. Whisper handles wav/mp3/m4a/ogg natively via ffmpeg, so no manual
format conversion is needed here.
"""

import os
import tempfile
import whisper
import streamlit as st


@st.cache_resource(show_spinner="Loading Whisper model (first run only)...")
def load_whisper_model():
    """
    Cached so the model loads only once per app session.
    "base" is a good accuracy/speed tradeoff for short concept explanations.
    Use "small" or "medium" for higher accuracy if you have the compute.
    """
    return whisper.load_model("base")


def transcribe_audio(audio_bytes: bytes, original_format: str = "wav") -> dict:
    """
    Args:
        audio_bytes: raw bytes of the uploaded/recorded audio file
        original_format: file extension Streamlit gave us (e.g. "wav", "mp3")

    Returns:
        dict with keys:
            success (bool)
            transcript (str)
            error (str | None)
    """
    model = load_whisper_model()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{original_format}") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = model.transcribe(tmp_path, fp16=False)
        transcript = result.get("text", "").strip()

        if not transcript:
            return {
                "success": False,
                "transcript": "",
                "error": "Could not detect any speech in the audio. Please try again.",
            }

        return {"success": True, "transcript": transcript, "error": None}

    except Exception as e:
        return {"success": False, "transcript": "", "error": f"Transcription failed: {e}"}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
