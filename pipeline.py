import os
from vad import trim_silence
from asr import transcribe
from diarize import diarize, DIARIZATION_ENABLED
from translate import translate_to_english


def _assign_speakers(asr_segments: list[dict], diar_segments: list[dict]) -> list[dict]:
    for seg in asr_segments:
        mid = (seg["start"] + seg["end"]) / 2
        speaker = "SPEAKER_00"
        for d in diar_segments:
            if d["start"] <= mid <= d["end"]:
                speaker = d["speaker"]
                break
        seg["speaker"] = speaker
    return asr_segments


def process_audio(audio_path: str) -> dict:
    trimmed_path = trim_silence(audio_path)
    asr_result = transcribe(trimmed_path)

    segments = asr_result["segments"]

    if DIARIZATION_ENABLED:
        diar_segments = diarize(trimmed_path)
        segments = _assign_speakers(segments, diar_segments)
    else:
        for seg in segments:
            seg["speaker"] = None

    for seg in segments:
        seg["translation_en"] = translate_to_english(
            seg["text"], asr_result["language"]
        )

    if trimmed_path != audio_path and os.path.exists(trimmed_path):
        os.unlink(trimmed_path)

    return {
        "language": asr_result["language"],
        "language_probability": asr_result["language_probability"],
        "diarization_enabled": DIARIZATION_ENABLED,
        "segments": segments,
    }
