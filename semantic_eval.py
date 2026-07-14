"""
semantic_eval.py
------------------
Semantic similarity computation using Sentence-BERT.

Covers Epic 2 / Story: "Semantic Understanding & Similarity Engine"
- Generates embeddings for student transcript + reference concept
- Computes cosine similarity between them
- Normalizes score into a consistent [0, 1] range
"""

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def load_sbert_model(model_name: str = DEFAULT_MODEL_NAME):
    """
    Load and cache the Sentence-BERT model once per session
    (Epic 4 / Story: Performance Testing and Optimization).
    """
    return SentenceTransformer(model_name)


def semantic_similarity(transcript: str, reference_concept: str,
                         model_name: str = DEFAULT_MODEL_NAME) -> float:
    """
    Compute the semantic (cosine) similarity between a student's spoken
    transcript and the reference concept text.

    Args:
        transcript: transcribed student explanation
        reference_concept: the ground-truth concept description
        model_name: SBERT model variant to use

    Returns:
        A normalized similarity score in [0.0, 1.0].
    """
    if not transcript or not reference_concept:
        return 0.0

    model = load_sbert_model(model_name)

    embeddings = model.encode(
        [transcript, reference_concept],
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    raw_score = util.cos_sim(embeddings[0], embeddings[1]).item()

    # cosine similarity is naturally in [-1, 1]; normalize to [0, 1]
    normalized_score = (raw_score + 1.0) / 2.0
    return float(np.clip(normalized_score, 0.0, 1.0))
