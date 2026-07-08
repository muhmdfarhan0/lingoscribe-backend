import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SAMPLE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.wav")


def test_imports():
    import asr
    import vad
    import diarize
    import translate
    import pipeline
    import main


def test_health():
    from main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="sample.wav not found")
def test_transcribe_structure():
    from main import app
    client = TestClient(app)
    with open(SAMPLE, "rb") as f:
        resp = client.post(
            "/transcribe",
            files={"file": ("sample.wav", f, "audio/wav")},
        )
    assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "language" in data
    assert "segments" in data
    assert isinstance(data["segments"], list)
    if data["segments"]:
        seg = data["segments"][0]
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg
        assert "translation_en" in seg
