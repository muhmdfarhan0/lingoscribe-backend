# LingoScribe Mini — Backend

A speech-AI inference pipeline for Pakistani local languages (Urdu, Punjabi).
Accepts audio uploads, strips silence, transcribes with Whisper, and translates
each segment to English using NLLB-200 — all running on CPU so it deploys
affordably on Render's free tier.

```
                        ┌──────────────────────────────────────────────┐
                        │               LingoScribe Pipeline            │
                        └──────────────────────────────────────────────┘
                                              │
                              POST /transcribe  (multipart audio)
                                              │
                                     ┌────────▼────────┐
                                     │   vad.py         │
                                     │  Silero-VAD      │
                                     │  trim silence    │
                                     └────────┬────────┘
                                              │ trimmed WAV
                                     ┌────────▼────────┐
                                     │   asr.py         │
                                     │  faster-whisper  │
                                     │  small / int8    │
                                     └────────┬────────┘
                                              │ segments + lang
                          ┌───────────────────┴──────────────────────┐
                          │ if ENABLE_DIARIZATION=true (flagged off)  │
                          │   diarize.py · pyannote · speaker labels  │
                          └───────────────────┬──────────────────────┘
                                              │ segments (± speaker)
                                     ┌────────▼────────┐
                                     │ translate.py     │
                                     │ NLLB-200 / CT2   │
                                     │ → English        │
                                     └────────┬────────┘
                                              │
                                        JSON response
```

## Setup (local)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### One-time NLLB model conversion

```bash
python scripts/convert_nllb.py
```

This downloads `facebook/nllb-200-distilled-600M` and converts it to
CTranslate2 int8 format (~600 MB). The resulting directory
`converted_models/nllb-200-distilled-600M/` should be committed via Git LFS
or downloaded at container startup (see `render.yaml` comments).

### Run locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t lingoscribe-backend .
docker run -p 8000:8000 -e PORT=8000 lingoscribe-backend
```

## API

### `GET /health`

Returns `{"status": "ok"}` — used by Render health checks.

### `POST /transcribe`

| Field | Type | Description |
|-------|------|-------------|
| `file` | multipart audio | WAV, MP3, OGG, FLAC, M4A accepted |

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

## Known limitations

- **CPU-only inference** — transcription of a 30-second clip takes ~15–60 s
  on Render free tier. A GPU instance would be 10–20× faster.
- **Diarization flagged off** — `pyannote/speaker-diarization-3.1` requires
  a HuggingFace gated-model approval and a HF token. Set
  `ENABLE_DIARIZATION=true` + `HF_TOKEN=<token>` to enable it locally.
- **NLLB translation is a demo** — the converted model must be present at
  `NLLB_MODEL_DIR`. On Render free tier without Git LFS the translation
  step returns a placeholder message rather than crashing the API.
- **LoRA fine-tune is a proof-of-concept** — `scripts/finetune_whisper.py`
  shows the training loop runs and loss decreases; it does not produce a
  usable model (20 samples, 30 steps).
- **No streaming** — audio is buffered and processed in full before returning;
  real-time streaming would require a WebSocket endpoint.

## What I'd add with more time / data / GPU

- Stream segments back via WebSocket as they're transcribed (reduce perceived latency)
- Fine-tune Whisper small on 10k+ labelled Urdu utterances (Common Voice + synthetic augmentation) on a GPU
- Enable diarization on a paid tier with a persistent volume for the pyannote model weights
- Add a language-detection override parameter so users can force `ur` / `pa` when auto-detection is uncertain
- Cache transcription results by audio hash to avoid redundant computation
- Rate limiting and auth token validation for the public API

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Port uvicorn binds to (Render sets this automatically) |
| `ALLOWED_ORIGIN` | `http://localhost:5500` | CORS allowed origin (set to your Vercel URL) |
| `NLLB_MODEL_DIR` | `converted_models/nllb-200-distilled-600M` | Path to converted NLLB model |
| `ENABLE_DIARIZATION` | `false` | Set `true` to enable speaker diarization |
| `HF_TOKEN` | *(unset)* | HuggingFace token — required only when diarization is enabled |
