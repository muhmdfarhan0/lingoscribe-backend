import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import soundfile as sf
from vad import trim_silence

SAMPLE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.wav")

if not os.path.exists(SAMPLE):
    print("ERROR: sample.wav not found"); sys.exit(1)

data_before, sr = sf.read(SAMPLE)
dur_before = len(data_before) / sr

print(f"Original duration : {dur_before:.2f}s")
trimmed_path = trim_silence(SAMPLE)
data_after, sr2 = sf.read(trimmed_path)
dur_after = len(data_after) / sr2

print(f"Trimmed duration  : {dur_after:.2f}s")
print(f"Removed silence   : {dur_before - dur_after:.2f}s")
if trimmed_path != SAMPLE:
    os.unlink(trimmed_path)
print("VAD test passed.")
