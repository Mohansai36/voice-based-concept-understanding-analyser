"""
app.py
------
Streamlit front-end and main application logic for the
Voice-Based Concept Understanding Analyser (VBCUA).

Covers Epic 3: UI Development
- User Interface Design and Visualization
- Input Handling and Session State Management
- Output Rendering and Report Generation
"""

import os
import streamlit as st

from audio_utils import save_uploaded_file, extract_audio_features, save_waveform
from speech_to_text import speech_to_text
from semantic_eval import semantic_similarity
from scoring_engine import filler_word_ratio, evaluate_understanding
from report_generator import generate_pdf_report


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VBCUA — Voice-Based Concept Understanding Analyser",
    page_icon="🎙️",
    layout="centered",
)

CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e5e7eb;
    }
    .result-banner {
        padding: 0.9rem 1.2rem;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization (Epic 3 / Story: Input Handling & Session State)
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "audio_path": None,
        "waveform_path": None,
        "transcript": None,
        "audio_features": None,
        "filler_ratio": None,
        "similarity": None,
        "final_score": None,
        "understanding_level": None,
        "level_color": None,
        "pdf_path": None,
        "analysis_done": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🎙️ Voice-Based Concept Understanding Analyser</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Automated evaluation of spoken conceptual explanations using AI.</div>',
            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reference concept input
# ---------------------------------------------------------------------------
st.subheader("Reference Concept")
reference_concept = st.text_area(
    "Enter the concept the student is expected to explain",
    placeholder="e.g. Machine Learning is a subset of artificial intelligence that allows "
                "systems to learn patterns from data and improve performance without being "
                "explicitly programmed.",
    height=100,
)

# ---------------------------------------------------------------------------
# Audio upload
# ---------------------------------------------------------------------------
st.subheader("Upload Student Audio (WAV)")
uploaded_file = st.file_uploader(
    "Drag and drop file here",
    type=["wav", "mp3"],
    help="Limit 200MB per file • WAV, MP3",
)

if uploaded_file is not None and st.session_state.audio_path is None:
    try:
        st.session_state.audio_path = save_uploaded_file(uploaded_file)
        st.session_state.waveform_path = save_waveform(st.session_state.audio_path)
    except Exception as exc:
        st.error(f"Could not process the uploaded audio file: {exc}")

if st.session_state.audio_path:
    st.audio(st.session_state.audio_path)
    if st.session_state.waveform_path:
        st.image(st.session_state.waveform_path, caption="Audio Waveform", use_container_width=True)
else:
    st.info("Upload an audio file to begin analysis.")

# Reset button — clears session state for a new evaluation
col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("🔄 Reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ---------------------------------------------------------------------------
# Analysis trigger
# ---------------------------------------------------------------------------
with col_b:
    analyze_clicked = st.button(
        "Analyze Concept Understanding",
        disabled=(st.session_state.audio_path is None or not reference_concept.strip()),
        type="primary",
    )

if analyze_clicked:
    if not reference_concept.strip():
        st.warning("Please enter a reference concept before analyzing.")
    elif not st.session_state.audio_path:
        st.warning("Please upload a student audio file before analyzing.")
    else:
        with st.spinner("Processing and evaluating..."):
            try:
                audio_path = st.session_state.audio_path

                transcript = speech_to_text(audio_path)
                audio_features = extract_audio_features(audio_path)
                filler_ratio = filler_word_ratio(transcript)
                similarity = semantic_similarity(transcript, reference_concept)
                score, level, color = evaluate_understanding(similarity, filler_ratio, audio_features)

                st.session_state.transcript = transcript
                st.session_state.audio_features = audio_features
                st.session_state.filler_ratio = filler_ratio
                st.session_state.similarity = similarity
                st.session_state.final_score = score
                st.session_state.understanding_level = level
                st.session_state.level_color = color
                st.session_state.analysis_done = True
                st.session_state.pdf_path = None  # invalidate any stale report

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.session_state.analysis_done = False


# ---------------------------------------------------------------------------
# Results rendering (Epic 3 / Story: Output Rendering and Report Generation)
# ---------------------------------------------------------------------------
if st.session_state.analysis_done:
    st.success("Analysis Completed")

    st.subheader("Transcribed Explanation")
    st.write(st.session_state.transcript or "_No speech detected in the audio._")

    st.markdown(
        f'<div class="result-banner" style="background-color:{st.session_state.level_color};">'
        f'Understanding Score: {st.session_state.final_score}/100 — {st.session_state.understanding_level}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Semantic Similarity", f"{st.session_state.similarity:.2f}")
    m2.metric("Filler Word Ratio", f"{st.session_state.filler_ratio:.2f}")
    m3.metric("Pause Ratio", f"{st.session_state.audio_features['pause_ratio']:.2f}")
    m4.metric("Confidence (Energy)", f"{st.session_state.audio_features['rms_energy']:.4f}")

    # --- Report generation ---
    st.subheader("Download Report")
    if st.button("📄 Generate PDF Report"):
        with st.spinner("Generating report..."):
            metrics = {
                "semantic_similarity": st.session_state.similarity,
                "filler_ratio": st.session_state.filler_ratio,
                "pause_ratio": st.session_state.audio_features["pause_ratio"],
                "rms_energy": st.session_state.audio_features["rms_energy"],
            }
            st.session_state.pdf_path = generate_pdf_report(
                reference_concept=reference_concept,
                transcript=st.session_state.transcript,
                waveform_image_path=st.session_state.waveform_path,
                metrics=metrics,
                final_score=st.session_state.final_score,
                understanding_level=st.session_state.understanding_level,
            )

    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=f,
                file_name=os.path.basename(st.session_state.pdf_path),
                mime="application/pdf",
            )
