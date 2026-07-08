import os
import subprocess
import tempfile
import httpx

GROQ_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

LANG_CODE_MAP = {
    "urdu": "ur", "punjabi": "pa", "hindi": "hi",
    "english": "en", "arabic": "ar", "french": "fr",
    "spanish": "es", "german": "de", "chinese": "zh",
    "japanese": "ja", "korean": "ko", "persian": "fa",
    "bengali": "bn", "sindhi": "sd", "pashto": "ps",
}

_GROQ_NATIVE = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".opus", ".flac"}


def _to_supported_format(audio_path: str) -> tuple[str, bool]:
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in _GROQ_NATIVE:
        return audio_path, False
    out = tempfile.mktemp(suffix=".mp3")
    subprocess.run(
        ["ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1", "-b:a", "64k", out, "-y"],
        check=True, capture_output=True,
    )
    return out, True


def transcribe(audio_path: str) -> dict:
    converted, is_temp = _to_supported_format(audio_path)
    try:
        with open(converted, "rb") as f:
            audio_bytes = f.read()
        filename = os.path.basename(converted)
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                GROQ_AUDIO_URL,
                headers={"Authorization": f"Bearer {GROQ_KEY}"},
                data={"model": "whisper-large-v3", "response_format": "verbose_json"},
                files={"file": (filename, audio_bytes)},
            )
            resp.raise_for_status()
        data = resp.json()
        lang = data.get("language", "").lower()
        lang_code = LANG_CODE_MAP.get(lang, lang[:2] if len(lang) >= 2 else "ur")
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in data.get("segments", [])
            if s.get("text", "").strip()
        ]
        return {
            "language": lang_code,
            "language_probability": 1.0,
            "segments": segments,
        }
    finally:
        if is_temp and os.path.exists(converted):
            os.unlink(converted)
