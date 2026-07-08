# LingoScribe — Backend

Speech intelligence API for Urdu and Punjabi. Accepts audio in any format, strips silence, transcribes with Whisper, translates each segment to English with NLLB-200, and surfaces AI-generated insights via Groq — all on CPU so it runs affordably on free-tier cloud hosting.

```
                    ┌──────────────────────────────────────┐
                    │          LingoScribe Pipeline         │
                    └──────────────────────────────────────┘
                                      │
                      POST /transcribe  (multipart audio)
                                      │
                             ┌────────▼────────┐
                             │    vad.py        │
                             │  Silero-VAD      │  strips silence
                             └────────┬────────┘
                                      │
                             ┌────────▼────────┐
                             │    asr.py        │
                             │ faster-whisper   │  small / int8 / CPU
                             └────────┬────────┘
                                      │  segments + language
                             ┌────────▼────────┐
                             │  translate.py    │
                             │  NLLB-200 / CT2  │  → English
                             └────────┬────────┘
                                      │
                                 JSON response

          POST /analyze  ──►  Groq openai/gpt-oss-120b  ──►  insights
          POST /ask      ──►  Groq openai/gpt-oss-120b  ──►  answer
```

## Tech stack

| Layer | Library | Notes |
|-------|---------|-------|
| API | FastAPI + uvicorn | async, multipart upload |
| ASR | faster-whisper (small, int8) | CPU inference, ~15-60s per 30s clip |
| VAD | silero-vad 6.x | soundfile backend, no torchaudio required |
| Translation | NLLB-200-distilled-600M via CTranslate2 | one-time offline conversion |
| Insights & Q&A | Groq `openai/gpt-oss-120b` | via `/analyze` and `/ask` endpoints |
| Container | python:3.11-slim + ffmpeg | Docker, deployed on Render |

## Quick start

```bash
git clone https://github.com/muhmdfarhan0/lingoscribe-backend
cd lingoscribe-backend

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

cp .env.example .env              # fill in GROQ_API_KEY

uvicorn main:app --reload --port 8000
```

### One-time NLLB model conversion (enables translation)

```bash
python scripts/convert_nllb.py
```

Downloads `facebook/nllb-200-distilled-600M` and converts it to CTranslate2 int8 format (~600 MB). Commit via Git LFS or download at container startup — see `render.yaml` for options.

## API reference

### `GET /health`
Returns `{"status": "ok"}`. Used by Render health checks.

### `POST /transcribe`
Upload audio via multipart form.

| Field | Type | Description |
|-------|------|-------------|
| `file` | `audio/*` | Any format: WAV, MP3, OGG, FLAC, M4A, WEBM, OPUS, AAC, WMA, MP4 |

**Response**
```json
{
  "language": "ur",
  "language_probability": 0.98,
  "diarization_enabled": false,
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "یہ ایک آزمائشی جملہ ہے",
      "speaker": null,
      "translation_en": "This is a test sentence."
    }
  ]
}
```

### `POST /analyze`
AI-generated insights for a transcript.

```json
// Request
{ "text": "English transcript text", "language": "ur" }

// Response
{
  "summary": "2-3 sentence summary",
  "key_topics": ["topic 1", "topic 2"],
  "tone": "conversational",
  "observation": "Notable speech pattern or context"
}
```

### `POST /ask`
Ask any question about a transcript.

```json
// Request
{ "transcript": "Full transcript text", "question": "What was discussed?" }

// Response
{ "answer": "Direct answer from the transcript." }
```

## Docker

```bash
docker build -t lingoscribe-backend .
docker run -p 8000:8000 -e PORT=8000 -e GROQ_API_KEY=your_key lingoscribe-backend
```

## Utility scripts

| Script | Purpose |
|--------|---------|
| `scripts/convert_nllb.py` | One-time NLLB model conversion (run locally before deploy) |
| `scripts/noise_augment.py` | Audio augmentation demo: GSM bandpass, noise, time-stretch |
| `scripts/finetune_whisper.py` | LoRA fine-tune demo on Urdu Common Voice (proves training loop runs) |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Port uvicorn binds to (Render sets this automatically) |
| `ALLOWED_ORIGIN` | `http://localhost:5500` | CORS origin — set to your Vercel URL |
| `GROQ_API_KEY` | — | Groq API key for `/analyze` and `/ask` |
| `NLLB_MODEL_DIR` | `converted_models/nllb-200-distilled-600M` | Path to converted NLLB model |
| `ENABLE_DIARIZATION` | `false` | Enable speaker diarization (requires HuggingFace gated model access) |
| `HF_TOKEN` | — | HuggingFace token — required only when `ENABLE_DIARIZATION=true` |

## Known limitations

- **CPU-only inference** — transcribing a 30-second clip takes 15-60 s on free-tier hosting. A GPU would be 10-20x faster.
- **Diarization is off by default** — `pyannote/speaker-diarization-3.1` requires HuggingFace gated-model approval. Set `ENABLE_DIARIZATION=true` and `HF_TOKEN` after approval.
- **Translation requires model conversion** — run `scripts/convert_nllb.py` once. Without it, the API returns a placeholder instead of crashing.
- **LoRA fine-tune is a proof of concept** — `scripts/finetune_whisper.py` demonstrates the training loop and loss decrease on a small synthetic dataset, not a production model.

## What would come next

- WebSocket streaming for real-time segment delivery
- GPU fine-tuning on 10k+ labelled Urdu utterances (Common Voice + synthetic augmentation)
- Diarization on a paid tier with persistent model volume
- Language detection override parameter for uncertain cases
- Response caching by audio hash

## Contact

- [GitHub](https://github.com/muhmdfarhan0)
- [LinkedIn](https://www.linkedin.com/in/muhammad-farhan07567)
- [Website](https://www.farhanai.online/contact)
