"""
Fine-tune openai/whisper-small on AI4Bharat Shrutilipi Odia ASR dataset.
  - Base model : openai/whisper-small  (Apache 2.0)
  - Dataset    : ai4bharat/shrutilipi, config="odia"
  - Trainer    : HuggingFace Seq2SeqTrainer
  - Metric     : WER (word error rate)
  - Output     : stt/whisper_odia/
"""

from __future__ import annotations
import sys
import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_config(cfg_path: str | Path | None = None) -> dict:
    if cfg_path is None:
        cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main(cfg_path: str | Path | None = None) -> None:
    from datasets import load_dataset, Audio, DatasetDict
    from transformers import (
        WhisperFeatureExtractor,
        WhisperTokenizer,
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    import evaluate

    cfg = load_config(cfg_path)
    stt_cfg = cfg["stt"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = (device.type == "cuda")
    print(f"Device: {device}  |  fp16: {use_fp16}")

    base_model = stt_cfg["base_model"]
    output_dir = str(ROOT / stt_cfg["output_dir"])

    # --- processor & model (no language= to avoid Odia not supported error) ---
    feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
    tokenizer = WhisperTokenizer.from_pretrained(base_model)
    processor = WhisperProcessor.from_pretrained(base_model)
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # --- dataset ---
    print("Loading AI4Bharat Shrutilipi Odia dataset...")
    raw = load_dataset("ai4bharat/shrutilipi", "odia")

    train_cols = raw["train"].column_names
    text_col = next((c for c in ("sentence", "text", "transcription") if c in train_cols), None)
    if text_col is None:
        raise ValueError(f"No transcript column found. Available: {train_cols}")
    print(f"Transcript column: '{text_col}'")

    def prepare(batch: dict) -> dict:
        try:
            decoder = batch["audio_filepath"]
            audio_samples = decoder.get_all_samples()
            # data shape: [channels, samples] — take first channel, convert to numpy
            array = audio_samples.data[0].numpy()
            sr = audio_samples.sample_rate
            batch["input_features"] = feature_extractor(
                array, sampling_rate=sr
            ).input_features[0]
            batch["labels"] = tokenizer(batch[text_col]).input_ids
        except Exception:
            batch["input_features"] = None
            batch["labels"] = None
        return batch

    remove_cols = raw["train"].column_names
    raw = raw.map(prepare, remove_columns=remove_cols, num_proc=1)
    raw = raw.filter(lambda x: x["input_features"] is not None and x["labels"] is not None)

    print(f"Dataset size after filtering: {len(raw['train'])} train samples")

    # --- eval split ---
    if "validation" in raw:
        eval_ds = raw["validation"]
    elif "test" in raw:
        eval_ds = raw["test"]
    else:
        n_eval = max(200, len(raw["train"]) // 20)
        eval_ds = raw["train"].select(range(len(raw["train"]) - n_eval, len(raw["train"])))
        train_ds = raw["train"].select(range(len(raw["train"]) - n_eval))
        raw = DatasetDict({"train": train_ds, "validation": eval_ds})

    print(f"Train: {len(raw['train'])}  |  Eval: {len(eval_ds)}")

    # --- WER metric ---
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred: Any) -> Dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    # --- training args ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=stt_cfg["batch_size"],
        per_device_eval_batch_size=stt_cfg["batch_size"],
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=stt_cfg["max_steps"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=use_fp16,
        bf16=False,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="none",
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=raw["train"],
        eval_dataset=eval_ds,
        processing_class=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"Starting Whisper fine-tuning for {stt_cfg['max_steps']} steps...")
    trainer.train()

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\nWhisper model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Odia ASR")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
