import os

DIARIZATION_ENABLED = os.getenv("ENABLE_DIARIZATION", "false").lower() == "true"

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise EnvironmentError("HF_TOKEN env var required for diarization")
        from pyannote.audio import Pipeline
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
    return _pipeline


def diarize(audio_path: str) -> list[dict]:
    # pyannote.audio requires HuggingFace gated-model approval.
    # Set ENABLE_DIARIZATION=true and supply HF_TOKEN once access is confirmed.
    if not DIARIZATION_ENABLED:
        raise NotImplementedError(
            "Diarization is disabled. Set ENABLE_DIARIZATION=true and "
            "supply HF_TOKEN after approving pyannote/speaker-diarization-3.1 "
            "on huggingface.co."
        )
    pipeline = _get_pipeline()
    result = pipeline(audio_path)
    return [
        {"speaker": turn.label, "start": round(segment.start, 3), "end": round(segment.end, 3)}
        for segment, _, turn in result.itertracks(yield_label=True)
    ]
