"""
report_generator.py
---------------------
PDF report generation using ReportLab.

Covers Epic 3 / Story: "Output Rendering and Report Generation"
Produces a downloadable PDF containing:
- Reference concept
- Student transcription
- Waveform visualization
- Evaluation summary table
- Qualitative feedback (Strong / Moderate / Poor)
"""

import os
import uuid
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)


def generate_pdf_report(reference_concept: str, transcript: str,
                         waveform_image_path: str, metrics: dict,
                         final_score: int, understanding_level: str,
                         output_dir: str = "reports") -> str:
    """
    Build a structured PDF evaluation report.

    Args:
        reference_concept: the ground-truth concept text
        transcript: the student's transcribed explanation
        waveform_image_path: path to the waveform PNG (from audio_utils.save_waveform)
        metrics: dict with keys like semantic_similarity, filler_ratio,
                 pause_ratio, rms_energy (confidence)
        final_score: overall score out of 100
        understanding_level: "Strong Understanding" / "Moderate Understanding" / "Poor Understanding"
        output_dir: directory to save the generated PDF

    Returns:
        Path to the generated PDF file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"vbcua_report_{uuid.uuid4().hex[:8]}.pdf"
    pdf_path = os.path.join(output_dir, file_name)

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=18, spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyStyle", parent=styles["BodyText"], leading=16,
    )

    level_colors = {
        "Strong Understanding": colors.HexColor("#2ecc71"),
        "Moderate Understanding": colors.HexColor("#f39c12"),
        "Poor Understanding": colors.HexColor("#e74c3c"),
    }
    level_color = level_colors.get(understanding_level, colors.grey)

    elements = []

    # --- Title ---
    elements.append(Paragraph("Voice-Based Concept Understanding Analyser", title_style))
    elements.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["Normal"],
    ))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc"), spaceAfter=10))

    # --- Reference Concept ---
    elements.append(Paragraph("Reference Concept", heading_style))
    elements.append(Paragraph(reference_concept or "N/A", body_style))

    # --- Student Transcription ---
    elements.append(Paragraph("Student Transcription", heading_style))
    elements.append(Paragraph(transcript or "N/A", body_style))

    # --- Waveform ---
    elements.append(Paragraph("Audio Visualization", heading_style))
    if waveform_image_path and os.path.exists(waveform_image_path):
        elements.append(Image(waveform_image_path, width=16 * cm, height=5.5 * cm))
    else:
        elements.append(Paragraph("Waveform image not available.", body_style))

    # --- Evaluation Summary Table ---
    elements.append(Paragraph("Evaluation Summary", heading_style))
    table_data = [
        ["Metric", "Value"],
        ["Semantic Similarity", f"{metrics.get('semantic_similarity', 0):.2f}"],
        ["Filler Word Ratio", f"{metrics.get('filler_ratio', 0):.2f}"],
        ["Pause Ratio", f"{metrics.get('pause_ratio', 0):.2f}"],
        ["Confidence (Energy)", f"{metrics.get('rms_energy', 0):.4f}"],
        ["Final Score", f"{final_score}/100"],
        ["Understanding Level", understanding_level],
    ]

    table = Table(table_data, colWidths=[8 * cm, 8 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
        ("BACKGROUND", (0, -1), (-1, -1), level_color),
        ("TEXTCOLOR", (0, -1), (-1, -1), colors.white),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)

    elements.append(Spacer(1, 16))
    elements.append(Paragraph(
        "This report is generated by VBCUA as an educational support tool. "
        "It is intended to complement, not replace, human evaluation.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey),
    ))

    doc.build(elements)
    return pdf_path
