"""
Output Rendering and Report Generation
Epic 3: Streamlit UI Implementation and User Interaction (Interactive Reporting)

Builds a downloadable PDF summary including waveform image, all evaluation
metrics, an AI-generated summary, and qualitative feedback.
"""

import os
import tempfile
from datetime import datetime
from fpdf import FPDF


def _safe(text) -> str:
    """
    The built-in PDF font (Helvetica) only supports Latin-1 characters.
    Whisper can occasionally output non-Latin scripts (e.g. if fed music
    or non-English audio), which would otherwise crash PDF generation.
    This replaces any unsupported character instead of failing.
    """
    if text is None:
        return ""
    text = str(text)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_ai_summary(concept: str, sim_result: dict, audio_result: dict) -> str:
    """
    Rule-based natural-language summary generator that reads the computed
    scores and produces a short human-readable paragraph -- the
    "AI-generated summary" shown in the dashboard and PDF report.
    """
    parts = [f"The speaker attempted to explain '{concept}'."]

    score = sim_result.get("score", 0)
    if score >= 75:
        parts.append("They demonstrated a strong grasp of the core concept, accurately covering the key ideas.")
    elif score >= 45:
        parts.append("They showed a moderate understanding of the concept, touching on some key ideas while missing others.")
    else:
        parts.append("Their explanation showed limited alignment with the core concept and would benefit from revision.")

    fluency = audio_result.get("fluency_score", 0)
    filler_count = audio_result.get("filler_count", 0)
    pause_ratio = audio_result.get("pause_ratio_pct", 0)

    if fluency >= 80:
        parts.append("Delivery was confident and fluent, with minimal hesitation.")
    elif fluency >= 60:
        parts.append(f"Delivery was reasonably fluent, with {filler_count} filler word(s) and {pause_ratio}% pauses.")
    else:
        parts.append(f"Delivery showed frequent pauses ({pause_ratio}%) and filler word usage ({filler_count} instances), suggesting more practice is needed for confident, clear speech.")

    return " ".join(parts)


def build_pdf_report(concept: str, transcript: str, sim_result: dict, audio_result: dict, waveform_png_bytes: bytes = None) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Voice Based Concept Understanding Report", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _safe(f"Concept: {concept}"), ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Transcribed Explanation:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, _safe(transcript) if transcript else "(no transcript)")
    pdf.ln(3)

    # Waveform image
    if waveform_png_bytes:
        tmp_img_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                tmp_img.write(waveform_png_bytes)
                tmp_img_path = tmp_img.name
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "Waveform Visualization:", ln=True)
            pdf.image(tmp_img_path, w=180)
            pdf.ln(3)
        finally:
            if tmp_img_path and os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Conceptual Understanding (Semantic Similarity)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Score: {sim_result.get('score', 0)}/100", ln=True)
    pdf.cell(0, 6, _safe(f"Verdict: {sim_result.get('verdict', '-')}"), ln=True)
    pdf.multi_cell(0, 6, _safe(f"Feedback: {sim_result.get('missing_hint', '-')}"))
    pdf.ln(3)

    if "error" not in audio_result:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Speech Fluency & Delivery Analysis", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, _safe(f"Fluency Score: {audio_result.get('fluency_score', 0)}/100 ({audio_result.get('fluency_verdict', '-')})"), ln=True)
        pdf.cell(0, 6, f"Pause Ratio: {audio_result.get('pause_ratio_pct', 0)}%  |  Pauses Detected: {audio_result.get('pause_count', 0)}", ln=True)
        pdf.cell(0, 6, f"Filler Words: {audio_result.get('filler_count', 0)} instances ({audio_result.get('filler_ratio_pct', 0)}% of words)", ln=True)
        breakdown = audio_result.get("filler_breakdown", {})
        if breakdown:
            pdf.cell(0, 6, _safe("Breakdown: " + ", ".join(f"{k}={v}" for k, v in breakdown.items())), ln=True)
        pdf.cell(0, 6, f"RMS Energy (mean): {audio_result.get('rms_energy_mean', 0)}", ln=True)
        pdf.cell(0, 6, f"Duration: {audio_result.get('duration_sec', 0)}s", ln=True)
        pdf.ln(3)

    # Final comprehension score
    final_score = round(sim_result.get("score", 0) * 0.7 + audio_result.get("fluency_score", 0) * 0.3, 1) if "error" not in audio_result else sim_result.get("score", 0)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Final Comprehension Score: {final_score}/100", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "AI-Generated Summary:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, _safe(generate_ai_summary(concept, sim_result, audio_result)))

    return bytes(pdf.output())
