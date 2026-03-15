"""
Subhadra pretraining loop.
  - Streaming IterableDataset with sliding-window 1024-token chunks (cycles corpus)
  - AdamW + cosine LR with warmup
  - Mixed precision (torch.amp — works on both CUDA and CPU)
  - Gradient clipping at 1.0
  - Checkpoint save/resume (keeps last 3)
  - WandB logging (optional, skipped gracefully if not installed)
"""

from __future__ import annotations
import sys
import math
import time
import yaml
import argparse
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.subhadra import SubhadraConfig, SubhadraForCausalLM
from tokenizer.odia_tokenizer import OdiaTokenizer


# ---------------------------------------------------------------------------
# Dataset — cycles corpus indefinitely so training can reach max_steps
# ---------------------------------------------------------------------------

class OdiaTextDataset(IterableDataset):
    """Streams tokenized sliding-window chunks of length `seq_len`, cycling forever."""

    def __init__(self, corpus_path: str | Path, tokenizer: OdiaTokenizer,
                 seq_len: int = 1024, cycle: bool = True) -> None:
        self.corpus_path = Path(corpus_path)
        self.tokenizer   = tokenizer
        self.seq_len     = seq_len
        self.cycle       = cycle

    def _iter_once(self) -> Iterator[dict]:
        buffer: list[int] = []
        with open(self.corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = self.tokenizer.encode(line, add_bos=True, add_eos=True)
                buffer.extend(ids)
                while len(buffer) >= self.seq_len + 1:
                    chunk  = buffer[: self.seq_len + 1]
                    buffer = buffer[self.seq_len:]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:],  dtype=torch.long)
                    yield {"input_ids": x, "labels": y}

    def __iter__(self) -> Iterator[dict]:
        while True:
            yield from self._iter_once()
            if not self.cycle:
                break


# ---------------------------------------------------------------------------
# LR schedule — cosine with linear warmup
# ---------------------------------------------------------------------------

def cosine_lr(step: int, warmup: int, max_steps: int, max_lr: float,
              min_lr: float = 1e-5) -> float:
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: SubhadraForCausalLM, optimizer: torch.optim.Optimizer,
                    step: int, loss: float, ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt_step{step:07d}.pt"
    torch.save({
        "step":            step,
        "loss":            loss,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    # Keep only the 3 most recent checkpoints
    for old in sorted(ckpt_dir.glob("ckpt_step*.pt"))[:-3]:
        old.unlink()
    print(f"  [ckpt] saved {path.name}  (loss={loss:.4f})")


def load_checkpoint(model: SubhadraForCausalLM, optimizer: torch.optim.Optimizer,
                    ckpt_dir: Path) -> int:
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    if not ckpts:
        return 0
    latest = ckpts[-1]
    print(f"Resuming from {latest.name}")
    ckpt = torch.load(latest, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt["step"])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg_path: str | Path | None = None) -> None:
    if cfg_path is None:
        cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    pt   = cfg["pretrain"]
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device      = torch.device(device_type)
    print(f"Device: {device}")

    # --- tokenizer ---
    tok = OdiaTokenizer(ROOT / "tokenizer" / "odia_spm.model")

    # --- model ---
    model_cfg = SubhadraConfig(**cfg["model"])
    model     = SubhadraForCausalLM(model_cfg).to(device)
    print(f"Model: {model.num_parameters(trainable_only=False)/1e6:.2f}M params")

    # --- optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pt["lr"],
        weight_decay=pt["weight_decay"],
        betas=tuple(pt["betas"]),
        fused=(device_type == "cuda"),   # fused kernel when on GPU
    )

    # --- resume ---
    ckpt_dir   = ROOT / pt["checkpoint_dir"]
    start_step = load_checkpoint(model, optimizer, ckpt_dir)

    # --- corpus ---
    corpus = ROOT / cfg["data"]["clean_dir"] / "odia_corpus.txt"
    if not corpus.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus}\nRun: python data/clean_data.py")

    dataset = OdiaTextDataset(corpus, tok, seq_len=model_cfg.max_seq_len, cycle=True)
    loader  = DataLoader(dataset, batch_size=pt["batch_size"], num_workers=0)

    # --- AMP scaler (no-op on CPU) ---
    use_amp = device_type == "cuda"
    scaler  = torch.amp.GradScaler(device_type, enabled=use_amp)

    # --- optional WandB ---
    use_wandb = False
    try:
        import wandb
        wandb.init(project="subhadra", config=cfg, resume="allow", id="pretrain")
        use_wandb = True
        print("WandB: enabled")
    except Exception:
        print("WandB: not available, skipping")

    # --- training loop ---
    step         = start_step
    running_loss = 0.0
    last_loss    = 0.0
    t0           = time.time()

    print(f"\nPretraining from step {step} → {pt['max_steps']}  "
          f"(target loss < {pt['target_loss']})\n")

    model.train()
    pbar = tqdm(total=pt["max_steps"], initial=step, unit="step", dynamic_ncols=True)

    for batch in loader:
        if step >= pt["max_steps"]:
            break

        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        # update LR
        lr = cosine_lr(step, pt["warmup_steps"], pt["max_steps"], pt["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type, enabled=use_amp):
            loss, _ = model(input_ids, labels=labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), pt["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        last_loss     = loss.item()
        running_loss += last_loss
        step         += 1
        pbar.update(1)

        # --- logging ---
        if step % pt["log_every"] == 0:
            avg_loss = running_loss / pt["log_every"]
            elapsed  = time.time() - t0
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
            print(f"step {step:6d} | loss={avg_loss:.4f} | lr={lr:.2e} | {elapsed:.1f}s")
            if use_wandb:
                import wandb
                wandb.log({"train/loss": avg_loss, "train/lr": lr}, step=step)
            running_loss = 0.0
            t0 = time.time()

            if avg_loss < pt["target_loss"]:
                print(f"\nTarget loss {pt['target_loss']} reached at step {step}. Done.")
                break

        # --- checkpoint ---
        if step % pt["save_every"] == 0:
            save_checkpoint(model, optimizer, step, last_loss, ckpt_dir)

    pbar.close()

    # final checkpoint
    save_checkpoint(model, optimizer, step, last_loss, ckpt_dir)
    print("\nPretraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain Subhadra SLM")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    train(args.config)
