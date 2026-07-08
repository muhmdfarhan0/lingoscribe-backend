import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import process_audio

SAMPLE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.wav")
result = process_audio(SAMPLE)
print(json.dumps(result, ensure_ascii=False, indent=2))
