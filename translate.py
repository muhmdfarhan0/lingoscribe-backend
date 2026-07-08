import os
import json
import httpx

GROQ_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"

LANG_NAMES = {
    "ur": "Urdu", "pa": "Punjabi", "hi": "Hindi",
    "en": "English", "ar": "Arabic", "fa": "Persian",
    "bn": "Bengali", "sd": "Sindhi", "ps": "Pashto",
}


def translate_segments(segments: list, language: str) -> list:
    if language == "en":
        for seg in segments:
            seg["translation_en"] = seg["text"]
        return segments

    texts = [seg.get("text", "") for seg in segments]
    if not any(t.strip() for t in texts):
        for seg in segments:
            seg["translation_en"] = ""
        return segments

    lang_name = LANG_NAMES.get(language, language.upper())
    prompt = (
        f"Translate each {lang_name} phrase to English. "
        "Return ONLY a JSON array of strings — one English translation per input, same order, no explanation.\n\n"
        f"Input: {json.dumps(texts, ensure_ascii=False)}"
    )

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                GROQ_CHAT_URL,
                headers={"Authorization": f"Bearer {GROQ_KEY}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_completion_tokens": 1200,
                },
            )
            resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw[raw.find("["):]
        if raw.endswith("```"):
            raw = raw[:raw.rfind("]") + 1]
        translations = json.loads(raw)
        for seg, trans in zip(segments, translations):
            seg["translation_en"] = str(trans).strip() if trans else ""
    except Exception:
        for seg in segments:
            seg["translation_en"] = "[translation unavailable]"

    return segments
