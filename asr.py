import os
from faster_whisper import WhisperModel

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model


def transcribe(audio_path: str) -> dict:
    model = _get_model()
    segments, info = model.transcribe(audio_path, beam_size=5)
    segment_list = [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
    ]
    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "text": " ".join(s["text"] for s in segment_list),
        "segments": segment_list,
    }
