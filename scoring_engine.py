"""
scoring_engine.py
------------------
Filler word detection + final understanding score calculation.

Covers Epic 2 / Story: "Audio Feature Extraction & Scoring Engine"
- filler_word_ratio(transcript): text-based filler detection
- evaluate_understanding(...): combines semantic similarity, filler
  ratio, and audio confidence metrics into a final score (0-100) and
  a qualitative classification (Strong / Moderate / Poor).
"""

import re
from typing import Tuple

# Common English filler words / hesitation markers.
FILLER_WORDS = {
    "um", "uh", "erm", "ah", "like", "you know", "sort of", "kind of",
    "basically", "actually", "literally", "so", "well", "i mean",
    "right", "okay", "hmm",
}


def filler_word_ratio(transcript: str) -> float:
    """
    Compute the ratio of filler words to total words in a transcript.

    Args:
        transcript: transcribed text

    Returns:
        filler_ratio in [0.0, 1.0] — 0.0 for empty/invalid transcripts.
    """
    if not transcript or not transcript.strip():
        return 0.0

    # Normalize and tokenize on word boundaries (keeps it dependency-light;
    # NLTK's tokenizer can be swapped in if punkt is downloaded).
    words = re.findall(r"[a-zA-Z']+", transcript.lower())
    total_words = len(words)

    if total_words == 0:
        return 0.0

    text_lower = transcript.lower()
    filler_count = 0
    for filler in FILLER_WORDS:
        # count multi-word fillers ("you know") and single-word fillers alike
        filler_count += len(re.findall(rf"\b{re.escape(filler)}\b", text_lower))

    ratio = filler_count / total_words
    return float(min(ratio, 1.0))


def filler_word_stats(transcript: str) -> dict:
    """
    Returns the full FILLER_WORD_STATS record matching the ERD:
    filler_word_count, total_words, filler_ratio
    """
    words = re.findall(r"[a-zA-Z']+", transcript.lower()) if transcript else []
    total_words = len(words)
    ratio = filler_word_ratio(transcript)
    filler_count = round(ratio * total_words) if total_words else 0

    return {
        "filler_word_count": filler_count,
        "total_words": total_words,
        "filler_ratio": ratio,
    }


def evaluate_understanding(similarity: float, filler_ratio: float,
                            audio: dict) -> Tuple[int, str, str]:
    """
    Combine semantic similarity, filler word usage, and audio confidence
    metrics into a final understanding score + qualitative classification.

    Args:
        similarity: semantic similarity score in [0, 1]
        filler_ratio: filler word ratio in [0, 1]
        audio: dict with at least 'pause_ratio' and 'rms_energy' keys

    Returns:
        (score, understanding_level, color_hex)
    """
    score = 0

    # Semantic alignment (max 50 pts)
    score += 50 if similarity > 0.7 else 30 if similarity > 0.4 else 10

    # Fluency / filler usage (max 20 pts)
    score += 20 if filler_ratio < 0.05 else 10

    # Pause behavior (max 15 pts)
    score += 15 if audio.get("pause_ratio", 1.0) < 0.25 else 5

    # Vocal confidence / energy (max 15 pts)
    score += 15 if audio.get("rms_energy", 0.0) > 0.01 else 5

    if score >= 80:
        return score, "Strong Understanding", "#2ecc71"
    elif score >= 50:
        return score, "Moderate Understanding", "#f39c12"

    return score, "Poor Understanding", "#e74c3c"
