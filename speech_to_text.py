"""
speech_to_text.py
------------------
Whisper-based transcription logic for VBCUA.

Covers Epic 2 / Story: "Speech-to-Text Module Development"
- Loads OpenAI Whisper model (cached for performance)
- Transcribes uploaded audio (WAV/MP3) into text
- Handles varied accents / recording quality gracefully
"""

import whisper
import streamlit as st


# NOTE: "base" is a good default trade-off between speed and accuracy for a
# classroom-scale prototype. Swap to "small"/"medium" for higher accuracy if
# GPU resources allow.
DEFAULT_MODEL_SIZE = "base"


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str = DEFAULT_MODEL_SIZE):
    """
    Load and cache the Whisper model so it's only loaded once per session
    (Epic 4 / Story: Performance Testing and Optimization).
    """
    model = whisper.load_model(model_size)
    return model


def speech_to_text(file_path: str, model_size: str = DEFAULT_MODEL_SIZE) -> str:
    """
    Transcribe an audio file to text using Whisper.

    Args:
        file_path: path to the WAV/MP3 audio file
        model_size: whisper model variant to use

    Returns:
        The transcribed text (stripped). Returns an empty string with a
        graceful fallback message if transcription fails or produces
        no speech.
    """
    try:
        model = load_whisper_model(model_size)
        result = model.transcribe(file_path, fp16=False)
        text = result.get("text", "").strip()

        if not text:
            return ""

        return text

    except Exception as exc:
        # Graceful fallback so the UI can surface a clear error instead
        # of crashing the whole analysis pipeline.
        raise RuntimeError(f"Transcription failed: {exc}") from exc
