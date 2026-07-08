# LingoScribe — Backend

FastAPI backend for Urdu and Punjabi speech intelligence. Receives audio in any format, transcribes via Groq Whisper large-v3, batch-translates segments to English with an LLM, and generates AI insights — all in a lightweight Docker container with no local ML models.

**Live frontend:** https://lingoscribe-frontend.vercel.app  
**How it works:** https://lingoscribe-frontend.vercel.app/how-it-works.html  
**Frontend repo:** https://github.com/muhmdfarhan0/lingoscribe-frontend

---

## Architecture

![How LingoScribe Works](how-it-works-diagram.png)

---

## Pipeline

```
POST /transcribe (audio file)
        │
        ▼
  ffmpeg (if format not Groq-native)
        │
        ▼
  Groq Whisper large-v3
  → segments with timestamps + language detection
        │
        ▼
  Groq GPT-OSS-120B (batch translate all segments in one call)
  → English translation per segment
        │
        ▼
  JSON response {language, segments[{start, end, text, translation_en}]}

POST /analyze  ──►  Groq GPT-OSS-120B  ──►  {summary, key_topics, tone, observation}
POST /ask      ──►  Groq GPT-OSS-120B  ──►  {answer}
GET  /health   ──►  {"status": "ok"}
```

## Tech stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| API | FastAPI + Uvicorn | Async, multipart upload, CORS |
| Transcription | Groq Whisper large-v3 | Speech-to-text with timestamps |
| Translation | Groq GPT-OSS-120B | Batch segment translation, one LLM call |
| Insights & Q&A | Groq GPT-OSS-120B | `/analyze` and `/ask` endpoints |
| Format conversion | ffmpeg | WMA and other non-Groq-native formats |
| Container | python:3.11-slim + ffmpeg | No ML models in image — fast builds |
| Hosting | Render (Docker) | Free tier, env secrets for API key |

## Quick start

```bash
git clone https://github.com/muhmdfarhan0/lingoscribe-backend
cd lingoscribe-backend

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env       # add your GROQ_API_KEY

uvicorn main:app --reload --port 8000
```

## API reference

### `GET /health`
```json
{"status": "ok"}
```

### `POST /transcribe`
Multipart form upload.

| Field | Type | Description |
|-------|------|-------------|
| `file` | audio/* | WAV, MP3, OGG, FLAC, M4A, WEBM, OPUS, AAC, WMA, MP4 |

**Response**
```json
{
  "language": "ur",
  "language_probability": 1.0,
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
```json
// Request
{"text": "English transcript", "language": "ur"}

// Response
{
  "summary": "2–3 sentence summary",
  "key_topics": ["topic 1", "topic 2"],
  "tone": "conversational",
  "observation": "Notable speech pattern"
}
```

### `POST /ask`
```json
// Request
{"transcript": "Full transcript text", "question": "What was discussed?"}

// Response
{"answer": "Direct answer from the transcript."}
```

## Docker

```bash
docker build -t lingoscribe-backend .
docker run -p 8000:8000 -e PORT=8000 -e GROQ_API_KEY=your_key lingoscribe-backend
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Port uvicorn binds to |
| `ALLOWED_ORIGIN` | `http://localhost:5500` | CORS origin — set to your Vercel URL |
| `GROQ_API_KEY` | — | Required for all endpoints except `/health` |
| `ENABLE_DIARIZATION` | `false` | Speaker diarization (requires HF gated model) |
| `HF_TOKEN` | — | HuggingFace token — only if diarization enabled |

## Why Groq instead of self-hosted Whisper

Self-hosted Whisper (small model, 244 MB weights) + PyTorch exceeded Render's free-tier 512 MB RAM limit on every transcription request. The container OOMed, Render returned 503 without CORS headers, and the browser showed "Failed to fetch". Switching to Groq's API dropped the image to under 100 MB, eliminated cold-start model loading, and upgraded transcription quality to Whisper large-v3.

## Contact

- [GitHub](https://github.com/muhmdfarhan0)
- [LinkedIn](https://www.linkedin.com/in/muhammad-farhan07567)
- [Website](https://www.farhanai.online/contact)
