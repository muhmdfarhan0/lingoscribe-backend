# VAD is no longer used in the pipeline — Groq Whisper handles silence natively.
# Kept for local experimentation only.

def trim_silence(audio_path: str) -> str:
    return audio_path
