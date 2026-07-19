
# Voice Based Concept Understanding Analyser (VBCUA)

An AI-powered Streamlit web app that evaluates how well a user understands
and explains a concept **through spoken communication**, combining:

- **Speech-to-Text** — OpenAI Whisper
- **Semantic Similarity** — Sentence-BERT embeddings (cosine similarity) to check conceptual understanding, not just keyword overlap
- **Audio Feature Extraction** — Librosa: filler word usage, pause ratio, RMS energy
- **Interactive Reporting** — waveform visualization, AI-generated summary, downloadable PDF report

## Project Structure

```
voice_concept_analyser/
├── app.py                     # Main Streamlit app (dashboard, session state, orchestration)
├── requirements.txt
├── modules/
│   ├── stt.py                 # Whisper-based transcription (Epic 2)
│   ├── similarity.py          # Sentence-BERT semantic scoring (Epic 2)
│   ├── audio_features.py      # Filler words / pause ratio / RMS energy + fluency score (Epic 2)
│   ├── viz.py                 # Waveform visualization (Epic 3)
│   └── report.py              # AI-generated summary + PDF report builder (Epic 3)
```
## References:

Python: https://www.python.org/downloads/

FastAPI: https://fastapi.tiangolo.com/

Streamlit: https://docs.streamlit.io/

Librosa: https://librosa.org/doc/latest/index.html

Whisper: https://github.com/openai/whisper

Sentence-BERT: https://www.sbert.net/docs/

ReportLab: https://www.reportlab.com/docs/reportlab-userguide.pdf

Visual Studio Code: https://code.visualstudio.com/

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **ffmpeg required** — Whisper and Librosa both use it to decode mp3/m4a/ogg.
> Windows: download from ffmpeg.org and add to PATH. Mac: `brew install ffmpeg`.
> Linux: `sudo apt install ffmpeg`.

## Run

```bash
streamlit run app.py
```

## How it Works

1. User selects a concept (or provides a custom reference explanation)
2. User uploads or records audio explaining that concept
3. On **Analyze**:
   - `stt.py` transcribes the audio with Whisper
   - `similarity.py` embeds the reference and transcript with Sentence-BERT, computes cosine similarity → **Conceptual Understanding Score** (Strong / Moderate / Poor Understanding)
   - `audio_features.py` analyzes the raw waveform for **filler word usage**, **pause ratio**, and **RMS energy** → **Fluency Score**
   - `viz.py` renders the waveform for the dashboard and report
4. Dashboard shows: transcript, understanding score, fluency score, waveform, pause analysis, filler word stats, RMS energy, an AI-generated summary, and a combined **Final Comprehension Score**
5. User can download a full PDF report containing all of the above, including the waveform image

## ER / Data Model

- **Session**: concept_name, reference_text, transcript, timestamp
- **UnderstandingResult**: score, verdict, hint — belongs to a Session
- **FluencyResult**: fluency_score, pause_ratio, filler_count, rms_energy — belongs to a Session

## Possible Extensions

- Persist sessions to a database to track a user's progress over time
- Swap Whisper "base" model for "small"/"medium" for higher accuracy
- Deploy to Streamlit Community Cloud for a shareable public link
