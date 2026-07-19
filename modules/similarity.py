"""
Semantic Understanding & Similarity Engine
Epic 2: Core Logic Development (Speech Understanding & Evaluation Engine)

Uses Sentence-BERT embeddings (via sentence-transformers) to compare the
user's spoken explanation against a reference concept explanation, so
paraphrased or reworded (but correct) answers still score well.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer, util


@st.cache_resource(show_spinner="Loading Sentence-BERT model (first run only)...")
def load_model():
    """Cached so the model loads once per session, not on every rerun."""
    return SentenceTransformer("all-MiniLM-L6-v2")  # a Sentence-BERT model


def compute_similarity(reference_text: str, user_text: str) -> dict:
    """
    Args:
        reference_text: the ideal/reference explanation of the concept
        user_text: transcript of what the user said

    Returns:
        dict with keys:
            score (float)     -- 0-100 conceptual understanding score
            verdict (str)     -- "Strong Understanding" | "Moderate Understanding" | "Poor Understanding"
            missing_hint (str)
    """
    if not user_text.strip():
        return {"score": 0.0, "verdict": "Poor Understanding", "missing_hint": "No speech detected — try recording again."}

    model = load_model()
    embeddings = model.encode([reference_text, user_text], convert_to_tensor=True)
    raw_score = util.cos_sim(embeddings[0], embeddings[1]).item()

    score = max(0.0, min(1.0, (raw_score + 1) / 2)) * 100
    score = round(score, 1)

    if score >= 75:
        verdict = "Strong Understanding"
        hint = "The explanation closely matches the core concept — well done."
    elif score >= 45:
        verdict = "Moderate Understanding"
        hint = "The explanation touches the concept but misses some key ideas."
    else:
        verdict = "Poor Understanding"
        hint = "The explanation deviates significantly from the core concept — review and try again."

    return {"score": score, "verdict": verdict, "missing_hint": hint}
