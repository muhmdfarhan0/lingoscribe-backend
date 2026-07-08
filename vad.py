import os
import tempfile
import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

_vad_model = None

SAMPLE_RATE = 16000


def _get_vad():
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
    return _vad_model


def _load_as_tensor(audio_path: str) -> torch.Tensor:
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        # simple linear resampling via numpy interpolation
        target_len = int(len(data) * SAMPLE_RATE / sr)
        data = np.interp(
            np.linspace(0, len(data), target_len),
            np.arange(len(data)),
            data,
        ).astype(np.float32)
    return torch.from_numpy(data)


def trim_silence(audio_path: str) -> str:
    model = _get_vad()
    wav = _load_as_tensor(audio_path)
    timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLE_RATE)

    if not timestamps:
        return audio_path

    start_sample = timestamps[0]["start"]
    end_sample   = timestamps[-1]["end"]
    trimmed = wav[start_sample:end_sample].numpy()

    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False,
        dir=os.path.dirname(os.path.abspath(audio_path)),
    )
    sf.write(tmp.name, trimmed, SAMPLE_RATE)
    return tmp.name
