# Voice-Based Concept Understanding Analyser (VBCUA)

AI-powered evaluation of spoken conceptual explanations — combines Whisper
transcription, Sentence-BERT semantic similarity, and audio delivery metrics
(pause ratio, RMS energy) into a single understanding score with a
downloadable PDF report.

## Project Structure

```
vbcua/
├── app.py                 # Streamlit front-end and main app logic
├── audio_utils.py          # Audio loading, feature extraction, waveform plot
├── speech_to_text.py        # Whisper-based transcription
├── semantic_eval.py         # Sentence-BERT similarity engine
├── scoring_engine.py        # Filler word detection + final scoring
├── report_generator.py      # PDF report generation (ReportLab)
└── requirements.txt
```

## Epic 1 — Environment Setup

### 1. Create and activate a virtual environment

```bash
python -m venv vbcu_env

# Windows
vbcu_env\Scripts\activate

# macOS / Linux
source vbcu_env/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Requires **Python 3.10+**. Whisper additionally requires `ffmpeg` on your
system PATH:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows: download from ffmpeg.org and add to PATH
```

### 3. Run the app

```bash
streamlit run app.py
```

You should see:

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

## How it works

1. Enter the **reference concept** the student is expected to explain.
2. Upload the student's **WAV/MP3 audio** response.
3. Click **Analyze Concept Understanding**. The pipeline:
   - Transcribes the audio (Whisper)
   - Extracts audio features: pause ratio, RMS energy, zero-crossing rate (Librosa)
   - Computes filler word ratio from the transcript
   - Computes semantic similarity between transcript and reference concept (Sentence-BERT)
   - Combines all signals into a final score (0–100) and classification:
     **Strong / Moderate / Poor Understanding**
4. Review metrics in the UI, then **Generate PDF Report** for a downloadable summary.

## Scoring Logic

| Component | Max Points | Rule |
|---|---|---|
| Semantic similarity | 50 | >0.7 → 50, >0.4 → 30, else 10 |
| Filler word ratio | 20 | <0.05 → 20, else 10 |
| Pause ratio | 15 | <0.25 → 15, else 5 |
| Confidence (RMS energy) | 15 | >0.01 → 15, else 5 |

**Classification:** ≥80 Strong · ≥50 Moderate · <50 Poor

## Notes on Performance (Epic 4)

- Whisper and Sentence-BERT models are loaded once via
  `st.cache_resource` to avoid reloading on every rerun.
- For faster transcription during development, keep the Whisper model
  size at `"base"`; move to `"small"`/`"medium"` for higher accuracy in
  production if you have GPU access.

## Future Enhancements

- Multi-concept evaluation
- Learner progress tracking over time
- Multilingual support
- Real-time voice input
- Adaptive, personalized feedback
