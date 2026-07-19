"""
Voice Based Concept Understanding Analyser (VBCUA)
Main Streamlit Application

Epic 1: Streamlit Application Initialization
Epic 3: User Interface Design, Input Handling & Session State, Output Rendering

Run with:
    streamlit run app.py
"""

import streamlit as st

from modules.stt import transcribe_audio
from modules.similarity import compute_similarity
from modules.audio_features import analyze_audio
from modules.viz import plot_waveform
from modules.report import build_pdf_report, generate_ai_summary

st.set_page_config(page_title="Voice Based Concept Understanding Analyser", page_icon="🎙️", layout="centered")

# ---------------------------------------------------------------------------
# Session State Initialization (Epic 3: Input Handling and Session State)
# ---------------------------------------------------------------------------
for key, default in [
    ("transcript", ""),
    ("similarity_result", None),
    ("audio_result", None),
    ("waveform_png", None),
    ("analyzed", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

REFERENCE_CONCEPTS = {
    "Machine Learning": (
        "Machine learning is a branch of artificial intelligence where systems learn patterns "
        "from data instead of being explicitly programmed with rules. Models are trained on "
        "example data, and they improve their predictions or decisions as they see more data, "
        "generalizing to new, unseen inputs."
    ),
    "Cloud Computing": (
        "Cloud computing is the delivery of computing services -- servers, storage, databases, "
        "networking, and software -- over the internet, allowing on-demand access to shared "
        "resources without owning physical infrastructure. It typically follows a pay-as-you-go "
        "model and scales elastically with demand."
    ),
    "Photosynthesis": (
        "Photosynthesis is the process where plants convert light energy into chemical energy "
        "stored in glucose. It happens in the chloroplasts using chlorophyll, taking in carbon "
        "dioxide and water while releasing oxygen as a byproduct."
    ),
    "Custom (type your own reference)": "",
}

st.title("🎙️ Voice Based Concept Understanding Analyser")
st.caption(
    "An AI-powered assessment tool combining speech-to-text (Whisper), semantic similarity "
    "(Sentence-BERT), and audio-based fluency analysis (Librosa) to evaluate spoken "
    "conceptual understanding."
)

# ---------------------------------------------------------------------------
# Input Handling
# ---------------------------------------------------------------------------
st.subheader("1. Choose a concept")
concept_choice = st.selectbox("Pick a concept to explain", list(REFERENCE_CONCEPTS.keys()))

if concept_choice == "Custom (type your own reference)":
    reference_text = st.text_area("Enter the reference/ideal explanation", height=100)
    concept_label = st.text_input("Concept name", value="Custom Concept")
else:
    reference_text = REFERENCE_CONCEPTS[concept_choice]
    concept_label = concept_choice
    with st.expander("View reference explanation"):
        st.write(reference_text)

st.subheader("2. Record or upload your explanation")
input_mode = st.radio("Input method", ["Upload audio file", "Record with microphone"], horizontal=True)

audio_bytes = None
audio_format = "wav"

if input_mode == "Upload audio file":
    uploaded_file = st.file_uploader("Upload an audio file (wav, mp3, m4a, ogg)", type=["wav", "mp3", "m4a", "ogg"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        audio_format = uploaded_file.name.split(".")[-1].lower()
else:
    mic_input = st.audio_input("Record your explanation")
    if mic_input is not None:
        audio_bytes = mic_input.read()
        audio_format = "wav"

if audio_bytes:
    st.audio(audio_bytes)

# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------
st.subheader("3. Analyze")
can_analyze = audio_bytes is not None and bool(reference_text.strip())
if not reference_text.strip():
    st.info("Select or enter a reference explanation above before analyzing.")

if st.button("Analyze My Explanation", type="primary", disabled=not can_analyze):
    with st.spinner("Transcribing speech with Whisper..."):
        stt_result = transcribe_audio(audio_bytes, audio_format)

    if not stt_result["success"]:
        st.error(stt_result["error"])
        st.session_state.analyzed = False
    else:
        st.session_state.transcript = stt_result["transcript"]

        with st.spinner("Scoring conceptual understanding (Sentence-BERT)..."):
            st.session_state.similarity_result = compute_similarity(reference_text, stt_result["transcript"])

        with st.spinner("Analyzing fluency: filler words, pauses, RMS energy..."):
            audio_result = analyze_audio(audio_bytes, stt_result["transcript"], audio_format)
            st.session_state.audio_result = audio_result

        if "error" not in audio_result:
            with st.spinner("Rendering waveform..."):
                st.session_state.waveform_png = plot_waveform(audio_result["waveform"], audio_result["sample_rate"])

        st.session_state.analyzed = True

# ---------------------------------------------------------------------------
# Interactive Dashboard (Epic 3: Interactive Reporting and Performance Review)
# ---------------------------------------------------------------------------
if st.session_state.analyzed:
    st.divider()
    st.subheader("📊 Evaluation Dashboard")

    sim = st.session_state.similarity_result
    aud = st.session_state.audio_result

    st.markdown("**Transcribed Explanation:**")
    st.info(st.session_state.transcript)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Conceptual Understanding Score", f"{sim['score']}/100", sim["verdict"])
        st.caption(sim["missing_hint"])
    with col2:
        if "error" not in aud:
            st.metric("Fluency Score", f"{aud['fluency_score']}/100", aud["fluency_verdict"])
        else:
            st.warning(aud["error"])

    if "error" not in aud:
        final_score = round(sim["score"] * 0.7 + aud["fluency_score"] * 0.3, 1)
        st.metric("🏁 Final Comprehension Score", f"{final_score}/100")

        st.markdown("**Waveform Visualization:**")
        st.image(st.session_state.waveform_png)

        st.markdown("**Pause Analysis:**")
        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("Pause Ratio", f"{aud['pause_ratio_pct']}%")
        pcol2.metric("Pause Count", aud["pause_count"])
        pcol3.metric("Duration", f"{aud['duration_sec']}s")

        st.markdown("**Filler Word Statistics:**")
        fcol1, fcol2 = st.columns(2)
        fcol1.metric("Filler Word Count", aud["filler_count"])
        fcol2.metric("Filler Word Ratio", f"{aud['filler_ratio_pct']}%")
        if aud["filler_breakdown"]:
            st.write(aud["filler_breakdown"])
        else:
            st.caption("No common filler words detected.")

        st.markdown("**RMS Energy (speaking clarity/confidence proxy):**")
        st.write(f"Mean: {aud['rms_energy_mean']}  |  Std Dev: {aud['rms_energy_std']}")

        st.markdown("**🤖 AI-Generated Summary:**")
        st.success(generate_ai_summary(concept_label, sim, aud))

        pdf_bytes = build_pdf_report(concept_label, st.session_state.transcript, sim, aud, st.session_state.waveform_png)
        st.download_button(
            "📄 Download Full Report (PDF)",
            data=pdf_bytes,
            file_name=f"concept_report_{concept_label.replace(' ', '_')}.pdf",
            mime="application/pdf",
        )

    if st.button("Start New Analysis"):
        st.session_state.transcript = ""
        st.session_state.similarity_result = None
        st.session_state.audio_result = None
        st.session_state.waveform_png = None
        st.session_state.analyzed = False
        st.rerun()
