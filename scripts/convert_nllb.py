"""
One-time offline conversion: facebook/nllb-200-distilled-600M -> CTranslate2 format.

Run this ONCE locally before starting the server:
    python scripts/convert_nllb.py

Output lands in converted_models/nllb-200-distilled-600M/ (~600MB on disk).
Options for the converted model:
  - Commit it with Git LFS (recommended if you have LFS set up)
  - Download at container startup (add a startup script to render.yaml)
  - Keep it local-only and skip translation in cloud deploy

The model is NOT downloaded or converted at server startup to keep cold-start
times acceptable on free-tier hosting.
"""
import os
import ctranslate2
from transformers import AutoTokenizer

SRC_MODEL = "facebook/nllb-200-distilled-600M"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "converted_models", "nllb-200-distilled-600M")

if __name__ == "__main__":
    print(f"Converting {SRC_MODEL} to CTranslate2 format...")
    print(f"Output: {os.path.abspath(OUT_DIR)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    converter = ctranslate2.converters.OpusMTConverter(SRC_MODEL)
    # Use NLLB converter (OpusMT base works for M2M/NLLB family)
    converter = ctranslate2.converters.TransformersConverter(SRC_MODEL)
    converter.convert(OUT_DIR, quantization="int8", force=True)

    import shutil
    size_mb = sum(
        os.path.getsize(os.path.join(d, f))
        for d, _, files in os.walk(OUT_DIR)
        for f in files
    ) / 1e6
    print(f"Done. Model size: {size_mb:.1f} MB")
    print("Recommendation:")
    if size_mb > 300:
        print("  > 300 MB — use Git LFS or download at container startup.")
    else:
        print("  Small enough to commit directly (consider Git LFS anyway).")
