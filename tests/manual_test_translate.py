import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from translate import translate_to_english

test_cases = [
    ("یہ ایک آزمائشی جملہ ہے۔", "ur"),
    ("میرا نام فرحان ہے۔", "ur"),
    ("This is already English.", "en"),
]

for text, lang in test_cases:
    result = translate_to_english(text, lang)
    print(f"[{lang}] {text}")
    print(f"  -> {result}")
    print()
