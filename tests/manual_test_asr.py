import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asr import transcribe

SAMPLE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.wav")

if not os.path.exists(SAMPLE):
    print(f"ERROR: sample file not found at {SAMPLE}")
    print("Place a WAV file at tests/fixtures/sample.wav before running this test.")
    sys.exit(1)

print(f"Running ASR on: {SAMPLE}")
result = transcribe(SAMPLE)
print(f"\nDetected language : {result['language']} (confidence: {result['language_probability']})")
print(f"Full text         : {result['text'][:200]}...")
print(f"\nSegments ({len(result['segments'])} total):")
for seg in result["segments"][:5]:
    print(f"  [{seg['start']:.1f}s -> {seg['end']:.1f}s]  {seg['text']}")
if len(result["segments"]) > 5:
    print(f"  ... and {len(result['segments']) - 5} more segments")
