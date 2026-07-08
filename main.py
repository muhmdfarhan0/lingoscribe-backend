import os
import json
import asyncio
import tempfile
import shutil
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from pipeline import process_audio

app = FastAPI(title="LingoScribe", version="1.0.0")

ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "http://localhost:5500")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        ALLOWED_ORIGIN,
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "null",
        # Vercel preview and production domains
        "https://lingoscribe-frontend.vercel.app",
    ],
    allow_origin_regex=r"https://lingoscribe.*\.vercel\.app",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,
)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "openai/gpt-oss-120b"
GROQ_KEY     = os.getenv("GROQ_API_KEY", "")

LANG_NAMES = {
    "ur": "Urdu", "pa": "Punjabi", "hi": "Hindi",
    "en": "English", "ar": "Arabic",
}


def _groq_complete(messages: list, max_tokens: int = 800) -> str:
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            json={"model": GROQ_MODEL, "messages": messages,
                  "temperature": 0.7, "max_completion_tokens": max_tokens},
        )
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


class AnalyzeRequest(BaseModel):
    text: str
    language: str

class AskRequest(BaseModel):
    transcript: str
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"upload{suffix}")
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        result = process_audio(tmp_path)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/analyze")
async def analyze_endpoint(req: AnalyzeRequest):
    lang_name = LANG_NAMES.get(req.language, req.language.upper())
    prompt = f"""You are analyzing an audio transcription originally in {lang_name}, translated to English.

Transcript:
{req.text[:3000]}

Return a JSON object with exactly these keys:
{{
  "summary": "2-3 sentence plain-English summary of what was discussed",
  "key_topics": ["topic 1", "topic 2", "topic 3"],
  "tone": "one of: conversational / professional / emotional / instructional / narrative",
  "observation": "one insightful sentence about the speech style, code-switching, or notable patterns"
}}

Return only valid JSON, no markdown fences."""

    try:
        raw = await asyncio.get_event_loop().run_in_executor(
            None, _groq_complete, [{"role": "user", "content": prompt}], 600
        )
        return json.loads(raw)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Analysis error: {str(exc)}")


@app.post("/ask")
async def ask_endpoint(req: AskRequest):
    prompt = f"""You have access to the following audio transcript:

{req.transcript[:3000]}

Answer this question about the transcript concisely and accurately:
{req.question}

Be direct and specific. If the answer is not in the transcript, say so clearly."""

    try:
        answer = await asyncio.get_event_loop().run_in_executor(
            None, _groq_complete, [{"role": "user", "content": prompt}], 400
        )
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ask error: {str(exc)}")
