from asr import transcribe
from diarize import diarize, DIARIZATION_ENABLED
from translate import translate_segments


def _assign_speakers(asr_segments: list, diar_segments: list) -> list:
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
    asr_result = transcribe(audio_path)
    segments = asr_result["segments"]

    if DIARIZATION_ENABLED:
        diar_segs = diarize(audio_path)
        segments = _assign_speakers(segments, diar_segs)
    else:
        for seg in segments:
            seg["speaker"] = None

    segments = translate_segments(segments, asr_result["language"])

    return {
        "language": asr_result["language"],
        "language_probability": asr_result["language_probability"],
        "diarization_enabled": DIARIZATION_ENABLED,
        "segments": segments,
    }
