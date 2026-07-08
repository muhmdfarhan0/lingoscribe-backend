"""
LoRA fine-tune demo for Whisper-tiny on a small Urdu dataset.

PURPOSE: This script exists to demonstrate that the training loop runs
end-to-end and that the loss decreases across steps. It is NOT intended
to produce a usable model — the dataset is tiny and the step count is
minimal. A real fine-tune would require 10,000+ labelled utterances and
GPU hours; see README "What I'd add with more time/data/GPU" for details.

Requires (install separately from main requirements):
    pip install peft datasets accelerate

Dataset: mozilla-foundation/common_voice_13_0 "ur" split.
Falls back to a tiny synthetic set if download fails or quota is exceeded.
"""
import os
import sys
import json

try:
    from datasets import load_dataset, Audio
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from peft import get_peft_model, LoraConfig, TaskType
    import torch
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install peft datasets accelerate")
    sys.exit(1)

MODEL_NAME  = "openai/whisper-tiny"
LANG        = "ur"
TASK        = "transcribe"
MAX_SAMPLES = 20
MAX_STEPS   = 30
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output", "whisper-lora-demo")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading processor from {MODEL_NAME}...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANG, task=TASK)


def load_data():
    try:
        print("Loading Common Voice Urdu (may require HF account for large splits)...")
        ds = load_dataset(
            "mozilla-foundation/common_voice_13_0", LANG,
            split=f"train[:{MAX_SAMPLES}]", trust_remote_code=True,
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds
    except Exception as e:
        print(f"Common Voice unavailable ({e}). Using synthetic fallback.")
        return None


def preprocess(batch):
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    batch["input_features"] = inputs.input_features[0]
    with processor.as_target_processor():
        labels = processor(batch["sentence"]).input_ids
    batch["labels"] = labels
    return batch


def main():
    dataset = load_data()

    if dataset is None:
        # Synthetic fallback: a few Urdu sentences with dummy audio
        import numpy as np
        sentences = [
            "میرا نام فرحان ہے۔",
            "یہ ایک آزمائشی جملہ ہے۔",
            "لاہور پاکستان کا ایک بڑا شہر ہے۔",
        ] * 7  # repeat to get ~20 samples

        def make_sample(text):
            audio_array = np.random.randn(16000).astype(np.float32) * 0.01
            inp = processor(audio_array, sampling_rate=16000, return_tensors="pt")
            with processor.as_target_processor():
                labels = processor(text).input_ids
            return {"input_features": inp.input_features[0], "labels": labels}

        data_list = [make_sample(s) for s in sentences[:MAX_SAMPLES]]
        print(f"Synthetic dataset: {len(data_list)} samples")
    else:
        dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
        data_list = list(dataset)
        print(f"Common Voice dataset: {len(data_list)} samples")

    print(f"\nLoading {MODEL_NAME}...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANG, task=TASK)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    from torch.utils.data import Dataset as TorchDataset
    import torch

    class ListDataset(TorchDataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            item = self.items[i]
            return {
                "input_features": torch.tensor(item["input_features"]),
                "labels": torch.tensor(item["labels"]),
            }

    train_ds = ListDataset(data_list)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=5,
        logging_steps=5,
        save_steps=MAX_STEPS,
        fp16=False,              # CPU training
        predict_with_generate=False,
        report_to="none",
        no_cuda=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    print(f"\nStarting LoRA fine-tune ({MAX_STEPS} steps, CPU)...")
    result = trainer.train()
    print(f"\nTraining complete. Final loss: {result.training_loss:.4f}")
    print(f"Logs saved to {OUTPUT_DIR}")

    log_history = trainer.state.log_history
    loss_steps = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    print("\nLoss trend:")
    for step, loss in loss_steps:
        bar = "█" * int(loss * 20)
        print(f"  step {step:3d}: {loss:.4f}  {bar}")


if __name__ == "__main__":
    main()
