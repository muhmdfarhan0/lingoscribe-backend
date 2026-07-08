import os
from transformers import AutoTokenizer
import ctranslate2

MODEL_DIR = os.getenv("NLLB_MODEL_DIR", "converted_models/nllb-200-distilled-600M")
FLORES_MAP = {
    "ur": "urd_Arab",
    "pa": "pan_Guru",
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ar": "arb_Arab",
}
TARGET_LANG = "eng_Latn"

_tokenizer = None
_translator = None


def _load():
    global _tokenizer, _translator
    if _tokenizer is None:
        src = "facebook/nllb-200-distilled-600M"
        _tokenizer = AutoTokenizer.from_pretrained(src)
        _translator = ctranslate2.Translator(MODEL_DIR, device="cpu")


def translate_to_english(text: str, source_lang: str) -> str:
    flores_src = FLORES_MAP.get(source_lang, f"{source_lang}_Arab")
    if flores_src == TARGET_LANG:
        return text
    if not os.path.isdir(MODEL_DIR):
        # Model not yet converted — return a clear placeholder so the
        # API still works before the user runs scripts/convert_nllb.py.
        return f"[translation unavailable — run scripts/convert_nllb.py first]"
    _load()
    _tokenizer.src_lang = flores_src
    tokens = _tokenizer.convert_ids_to_tokens(_tokenizer.encode(text))
    results = _translator.translate_batch(
        [tokens],
        target_prefix=[[TARGET_LANG]],
        max_decoding_length=256,
    )
    translated_tokens = results[0].hypotheses[0][1:]  # strip target lang token
    return _tokenizer.decode(_tokenizer.convert_tokens_to_ids(translated_tokens))
