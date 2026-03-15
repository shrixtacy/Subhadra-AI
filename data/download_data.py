"""
Download Odia text corpora (updated dataset IDs for datasets >= 2.x):
  1. wikimedia/wikipedia  — Odia Wikipedia (20231101.or)
  2. ai4bharat/sangraha   — verified/ori  (largest high-quality Odia corpus)
  3. ai4bharat/sangraha   — unverified/ori

Saves raw text to data/raw/
Run: python data/download_data.py
"""

from __future__ import annotations
import yaml
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _stream_to_file(dataset_id: str, config: str | None,
                    split: str, text_field: str,
                    out_file: Path, max_rows: int | None = None) -> int:
    """Stream a HuggingFace dataset and write text lines to out_file."""
    kwargs: dict = dict(split=split, streaming=True)
    if config:
        kwargs["name"] = config
    ds = load_dataset(dataset_id, **kwargs)

    written = 0
    with open(out_file, "w", encoding="utf-8") as f:
        pbar = tqdm(ds, desc=f"  {out_file.stem}")
        for row in pbar:
            text = row.get(text_field, "").strip()
            if text:
                f.write(text + "\n")
                written += 1
            if max_rows and written >= max_rows:
                break
            pbar.set_postfix(rows=written)
    return written


def download_wikipedia(raw_dir: Path) -> Path:
    out = raw_dir / "wikipedia_or.txt"
    if out.exists():
        print(f"  Already exists: {out.name}")
        return out
    print("Downloading Odia Wikipedia (wikimedia/wikipedia 20231101.or)...")
    n = _stream_to_file("wikimedia/wikipedia", "20231101.or", "train", "text", out)
    print(f"  {n:,} articles → {out.name}")
    return out


def download_sangraha_verified(raw_dir: Path) -> Path:
    out = raw_dir / "sangraha_verified_or.txt"
    if out.exists():
        print(f"  Already exists: {out.name}")
        return out
    print("Downloading ai4bharat/sangraha verified (Odia split)...")
    # split='ori' is the Odia language split in sangraha
    n = _stream_to_file("ai4bharat/sangraha", "verified", "ori", "text", out)
    print(f"  {n:,} docs → {out.name}")
    return out


def download_sangraha_unverified(raw_dir: Path) -> Path:
    out = raw_dir / "sangraha_unverified_or.txt"
    if out.exists():
        print(f"  Already exists: {out.name}")
        return out
    print("Downloading ai4bharat/sangraha unverified (Odia split)...")
    # Cap at 2M rows to keep disk usage reasonable
    n = _stream_to_file("ai4bharat/sangraha", "unverified", "ori", "text", out,
                        max_rows=2_000_000)
    print(f"  {n:,} docs → {out.name}")
    return out


def main() -> None:
    cfg     = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_wikipedia(raw_dir)
    download_sangraha_verified(raw_dir)
    download_sangraha_unverified(raw_dir)

    files = list(raw_dir.glob("*.txt"))
    total = sum(p.stat().st_size for p in files)
    print(f"\nDone. {len(files)} files, {total/1e6:.1f} MB in {raw_dir}")


if __name__ == "__main__":
    main()
