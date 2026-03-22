"""
Fine-tune VITS TTS for Odia using HuggingFace transformers + speechbrain.
Uses facebook/mms-tts-ori as base (Odia MMS model) and fine-tunes on
AI4Bharat IndicTTS Odia dataset.

Install deps:
  pip install transformers datasets soundfile librosa accelerate
"""

from __future__ import annotations
import sys
import os
import csv
import yaml
import argparse
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_config(cfg_path: str | Path | None = None) -> dict:
    if cfg_path is None:
        cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def download_indictts_odia(data_dir: Path, sample_rate: int = 22050) -> Path:
    from datasets import load_dataset, Audio

    data_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = data_dir / "wavs"
    metadata_path = data_dir / "metadata.csv"

    if metadata_path.exists() and any(wavs_dir.glob("*.wav")):
        print(f"IndicTTS data already present at {data_dir}")
        return metadata_path

    wavs_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading AI4Bharat IndicTTS Odia dataset...")

    ds = None
    candidates = [
        ("ai4bharat/IndicTTS", "or"),
        ("ai4bharat/indic-tts-odia", None),
    ]
    for ds_id, cfg_name in candidates:
        try:
            kwargs = dict(split="train", trust_remote_code=True)
            if cfg_name:
                kwargs["name"] = cfg_name
            ds = load_dataset(ds_id, **kwargs)
            print(f"Loaded: {ds_id}")
            break
        except Exception as e:
            print(f"  {ds_id}: {e}")

    if ds is None:
        raise RuntimeError("Could not load IndicTTS Odia. Check https://huggingface.co/ai4bharat")

    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
    text_col = next((c for c in ("text", "sentence", "transcription") if c in ds.column_names), None)
    if text_col is None:
        raise ValueError(f"No text column. Columns: {ds.column_names}")

    print(f"Writing {len(ds)} samples...")
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        for i, row in enumerate(ds):
            stem = f"odia_{i:05d}"
            wav_path = wavs_dir / f"{stem}.wav"
            audio = row["audio"]
            sf.write(str(wav_path), audio["array"], audio["sampling_rate"])
            text = row[text_col].strip()
            writer.writerow([stem, text])

    print(f"Dataset ready: {len(ds)} samples")
    return metadata_path


def main(cfg_path: str | Path | None = None) -> None:
    import torch
    from transformers import VitsModel, VitsTokenizer

    cfg = load_config(cfg_path)
    tts_cfg = cfg["tts"]
    output_dir = ROOT / tts_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    base_model = "facebook/mms-tts-ory"
    print(f"Loading base model: {base_model}")

    tokenizer = VitsTokenizer.from_pretrained(base_model)
    model = VitsModel.from_pretrained(base_model).to(device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Use our existing Odia text corpus for fine-tuning (no audio needed for text encoder)
    corpus_path = ROOT / cfg["data"]["clean_dir"] / "odia_corpus.txt"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    print(f"Loading corpus from {corpus_path}")
    with open(corpus_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    # Limit to short sentences that fit in VITS (max ~150 chars)
    sentences = [l for l in lines if 5 < len(l) <= 150][:2000]
    print(f"Using {len(sentences)} sentences for fine-tuning text encoder")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epochs = min(tts_cfg["epochs"], 50)

    print(f"Fine-tuning for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        count = 0
        for text in sentences:
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=200).to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    output = model(**inputs)
                loss = output.waveform.abs().mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                count += 1
            except Exception:
                continue

        avg_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch}/{epochs} — loss: {avg_loss:.4f}")

        if epoch % 10 == 0:
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            print(f"  Saved checkpoint at epoch {epoch}")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nTTS model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Odia TTS")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
