"""
Waveform Visualization
Epic 3: Streamlit UI Implementation (Interactive Reporting)

Renders the raw audio waveform as an image, used both in the live
Streamlit dashboard and embedded into the downloadable PDF report.
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering, required for Streamlit/server contexts
import matplotlib.pyplot as plt


def plot_waveform(y: np.ndarray, sr: int):
    """
    Returns a PNG image (as bytes) of the waveform, suitable for
    st.image() or embedding into a PDF via a temp file.
    """
    fig, ax = plt.subplots(figsize=(8, 2.2))
    times = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(times, y, linewidth=0.5, color="#4C72B0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.set_xlim(0, max(times) if len(times) else 1)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
