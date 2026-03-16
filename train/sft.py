"""
Supervised Fine-Tuning (SFT) on Odia instruction dataset.
  - Loads latest pretrain checkpoint (warns if missing)
  - Freezes embedding layer
  - Fine-tunes 3 epochs at lr=1e-5
  - Only computes loss on the answer portion (prompt tokens masked with -100)
  - Saves SFT checkpoint per epoch + final best
"""

from __future__ import annotations
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.subhadra import SubhadraConfig, SubhadraForCausalLM
from tokenizer.odia_tokenizer import OdiaTokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """
    Each sample is a chat turn:
      <bos>Human: {question}<sep>Subhadra: {answer}<eos>
    Loss is computed only on the answer tokens (prompt masked with -100).
    """

    def __init__(self, jsonl_path: str | Path, tokenizer: OdiaTokenizer,
                 max_len: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.samples: List[Dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        print(f"SFTDataset: {len(self.samples)} examples loaded from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        ids  = self.tokenizer.encode_chat(item["question"], item["answer"])
        ids  = ids[: self.max_len]

        # Mask prompt tokens — only predict answer (everything after <sep>)
        sep_id  = self.tokenizer.sep_id
        sep_pos = (ids.index(sep_id) + 1) if sep_id in ids else 0
        labels  = [-100] * sep_pos + ids[sep_pos:]

        # Pad to max_len
        pad_len = self.max_len - len(ids)
        ids    += [self.tokenizer.pad_id] * pad_len
        labels += [-100] * pad_len

        return {
            "input_ids": torch.tensor(ids[:self.max_len],    dtype=torch.long),
            "labels":    torch.tensor(labels[:self.max_len], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_pretrain_ckpt(model: SubhadraForCausalLM, ckpt_dir: Path) -> None:
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    if not ckpts:
        print("WARNING: No pretrain checkpoint found — fine-tuning from random init.")
        return
    latest = ckpts[-1]
    print(f"Loading pretrain checkpoint: {latest.name}")
    ckpt = torch.load(latest, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])


def save_sft_ckpt(model: SubhadraForCausalLM, epoch: int,
                  loss: float, ckpt_dir: Path) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"sft_epoch{epoch:02d}.pt"
    torch.save({"epoch": epoch, "loss": loss, "model_state": model.state_dict()}, path)
    print(f"  [ckpt] saved {path.name}  (loss={loss:.4f})")
    return path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sft(cfg_path: str | Path | None = None) -> None:
    if cfg_path is None:
        cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    sft = cfg["sft"]
    sft["lr"] = float(sft["lr"])
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device      = torch.device(device_type)
    print(f"Device: {device}")

    # --- tokenizer ---
    tok = OdiaTokenizer(ROOT / "tokenizer" / "odia_spm.model")

    # --- model ---
    model_cfg = SubhadraConfig(**cfg["model"])
    model     = SubhadraForCausalLM(model_cfg).to(device)

    # --- load pretrained weights ---
    load_pretrain_ckpt(model, ROOT / cfg["pretrain"]["checkpoint_dir"])

    # --- freeze embedding ---
    model.embed.weight.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Embedding frozen. Trainable params: {trainable/1e6:.2f}M")

    # --- SFT data (auto-generate if missing) ---
    data_path = ROOT / sft["data_path"]
    if not data_path.exists():
        print(f"SFT data not found — generating at {data_path} ...")
        sys.path.insert(0, str(ROOT / "data"))
        from build_sft_data import build_dataset
        build_dataset(data_path)

    dataset = SFTDataset(data_path, tok, max_len=model_cfg.max_seq_len // 2)
    loader  = DataLoader(dataset, batch_size=sft["batch_size"],
                         shuffle=True, num_workers=0, drop_last=False)

    # --- optimizer (only trainable params) ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=sft["lr"],
        weight_decay=0.01,
    )

    # --- AMP (no-op on CPU) ---
    use_amp = device_type == "cuda"
    scaler  = torch.amp.GradScaler(device_type, enabled=use_amp)

    ckpt_dir  = ROOT / sft["checkpoint_dir"]
    best_loss = float("inf")
    best_path = None

    # --- epoch loop ---
    for epoch in range(1, sft["epochs"] + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{sft['epochs']}", unit="batch")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type, enabled=use_amp):
                loss, _ = model(input_ids, labels=labels)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{sft['epochs']} — avg loss: {avg_loss:.4f}")

        path = save_sft_ckpt(model, epoch, avg_loss, ckpt_dir)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = path

    # --- save best as final ---
    if best_path is not None:
        final = ckpt_dir / "sft_final.pt"
        import shutil
        shutil.copy(best_path, final)
        print(f"\nBest checkpoint (loss={best_loss:.4f}) copied to {final}")

    print("SFT complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT fine-tune Subhadra")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    train_sft(args.config)
