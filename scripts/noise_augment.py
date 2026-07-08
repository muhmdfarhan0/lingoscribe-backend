"""
Data augmentation demo: simulates real-world degradations common with
low-resource Pakistani language audio (GSM telephony, street noise, etc.).

Requires: audiomentations, matplotlib, numpy, soundfile
Install:  pip install audiomentations matplotlib

Output lands in scripts/output/ (gitignored).
A couple of example PNGs are committed for the README.
"""
import os
import sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from audiomentations import Compose, BandPassFilter, AddGaussianNoise, TimeStretch
except ImportError:
    print("Run: pip install audiomentations")
    sys.exit(1)

INPUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "sample.wav")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_RATE = 16000

augment = Compose([
    # Simulate GSM codec narrow-band response (300–3400 Hz passband)
    BandPassFilter(min_center_freq=300, max_center_freq=3400, p=1.0),
    # Street / background noise
    AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.015, p=1.0),
    # Slight speaking-rate variation
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
])


def plot_waveform(data, sr, title, path):
    fig, ax = plt.subplots(figsize=(10, 2))
    t = np.linspace(0, len(data) / sr, num=len(data))
    ax.plot(t, data, linewidth=0.5, color="#6c63ff")
    ax.set_title(title); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    if not os.path.exists(INPUT):
        print(f"ERROR: {INPUT} not found"); sys.exit(1)

    data, sr = sf.read(INPUT, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono

    print(f"Input: {len(data)/sr:.2f}s @ {sr}Hz")

    augmented = augment(samples=data, sample_rate=sr)

    before_wav = os.path.join(OUT_DIR, "before_augment.wav")
    after_wav  = os.path.join(OUT_DIR, "after_augment.wav")
    sf.write(before_wav, data, sr)
    sf.write(after_wav,  augmented, sr)
    print(f"Saved audio: {before_wav}")
    print(f"Saved audio: {after_wav}")

    plot_waveform(data,      sr, "Before augmentation", os.path.join(OUT_DIR, "before_waveform.png"))
    plot_waveform(augmented, sr, "After augmentation",  os.path.join(OUT_DIR, "after_waveform.png"))

    print("\nDone. Files are in scripts/output/ (gitignored).")
    print("Commit before_waveform.png and after_waveform.png for the README.")


if __name__ == "__main__":
    main()
