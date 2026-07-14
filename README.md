# Voice-Based Concept Understanding Analyser (VBCUA)

AI-powered evaluation of spoken conceptual explanations — combines Whisper
transcription, Sentence-BERT semantic similarity, and audio delivery metrics
(pause ratio, RMS energy) into a single understanding score with a
downloadable PDF report.

## Project Structure

```text
vbcua/
├── app.py                 # Streamlit front-end and main app logic
├── audio_utils.py         # Audio loading, feature extraction, waveform plot
├── speech_to_text.py      # Whisper-based transcription
├── semantic_eval.py       # Sentence-BERT similarity engine
├── scoring_engine.py      # Filler word detection + final scoring
├── report_generator.py    # PDF report generation (ReportLab)
└── requirements.txt
```

## Environment Setup

### Create and activate a virtual environment

```bash
python -m venv vbcu_env

# Windows
vbcu_env\Scripts\activate

# macOS / Linux
source vbcu_env/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Install FFmpeg and ensure it is available in your system PATH.

### Run the application

```bash
streamlit run app.py
```

## How It Works

1. Enter the reference concept.
2. Upload a student's WAV or MP3 recording.
3. The application:
   - Transcribes speech using Whisper.
   - Extracts audio features using Librosa.
   - Computes semantic similarity using Sentence Transformers.
   - Detects filler words.
   - Calculates an understanding score.
4. Download the generated PDF report.

## Future Enhancements

- Real-time microphone input
- Multilingual support
- Student history dashboard
- Personalized AI feedback