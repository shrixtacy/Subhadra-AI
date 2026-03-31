"""
Download multilingual corpora for Subhadra:
  - Hindi Wikipedia + IndicCorp
  - English Wikipedia (subset)
  - Indian mythology & folktales (multiple HuggingFace datasets)

Saves to data/raw/ and data/clean/

Run: python data/download_multilingual_data.py
"""

from __future__ import annotations
import re
import unicodedata
import yaml
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

ROOT = Path(__file__).parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Generic streaming downloader
# ---------------------------------------------------------------------------

def stream_to_file(
    dataset_id: str,
    config: str | None,
    split: str,
    text_field: str,
    out_file: Path,
    max_rows: int | None = None,
    extra_kwargs: dict | None = None,
) -> int:
    if out_file.exists():
        print(f"  Already exists: {out_file.name}")
        return 0
    kwargs: dict = dict(split=split, streaming=True)
    if config:
        kwargs["name"] = config
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    ds = load_dataset(dataset_id, **kwargs)
    written = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc=f"  {out_file.stem}"):
            text = row.get(text_field, "").strip()
            if text:
                f.write(text + "\n")
                written += 1
            if max_rows and written >= max_rows:
                break
    print(f"  {written:,} rows → {out_file.name}")
    return written


# ---------------------------------------------------------------------------
# Hindi corpora
# ---------------------------------------------------------------------------

def download_hindi_wikipedia(raw_dir: Path) -> None:
    print("\nDownloading Hindi Wikipedia...")
    stream_to_file(
        "wikimedia/wikipedia", "20231101.hi", "train", "text",
        raw_dir / "wikipedia_hi.txt",
    )


def download_hindi_sangraha(raw_dir: Path) -> None:
    print("\nDownloading ai4bharat/sangraha Hindi (verified)...")
    stream_to_file(
        "ai4bharat/sangraha", "verified", "hin", "text",
        raw_dir / "sangraha_verified_hi.txt",
        max_rows=1_000_000,
    )


# ---------------------------------------------------------------------------
# English corpora (capped — we only need enough for cross-lingual grounding)
# ---------------------------------------------------------------------------

def download_english_wikipedia(raw_dir: Path) -> None:
    print("\nDownloading English Wikipedia (capped 500k articles)...")
    stream_to_file(
        "wikimedia/wikipedia", "20231101.en", "train", "text",
        raw_dir / "wikipedia_en.txt",
        max_rows=500_000,
    )


def download_english_c4(raw_dir: Path) -> None:
    print("\nDownloading C4 English (capped 1M rows)...")
    stream_to_file(
        "allenai/c4", "en", "train", "text",
        raw_dir / "c4_en.txt",
        max_rows=1_000_000,
    )


def download_english_openwebtext(raw_dir: Path) -> None:
    print("\nDownloading OpenWebText (capped 500k rows)...")
    stream_to_file(
        "Skylion007/openwebtext", None, "train", "text",
        raw_dir / "openwebtext_en.txt",
        max_rows=500_000,
    )


# ---------------------------------------------------------------------------
# Indian Mythology & Folktales
# ---------------------------------------------------------------------------

def download_mythology_datasets(raw_dir: Path) -> None:
    """
    Attempt several HuggingFace datasets that contain Indian mythology,
    epics, and folktale content. Falls back gracefully if unavailable.
    """
    candidates = [
        # Ramayana / Mahabharata text
        {
            "id": "rahular/ramayana",
            "config": None, "split": "train", "field": "text",
            "out": "ramayana_en.txt",
        },
        {
            "id": "rahular/mahabharata",
            "config": None, "split": "train", "field": "text",
            "out": "mahabharata_en.txt",
        },
        # Panchatantra / Jataka tales
        {
            "id": "storytelling/indian_folktales",
            "config": None, "split": "train", "field": "story",
            "out": "indian_folktales_en.txt",
        },
        # Vedic / Puranic texts (Project Gutenberg mirror on HF)
        {
            "id": "storytelling/puranas",
            "config": None, "split": "train", "field": "text",
            "out": "puranas_en.txt",
        },
        # Hindi mythology
        {
            "id": "ai4bharat/sangraha", "config": "verified",
            "split": "hin", "field": "text",
            "out": "mythology_hi_sangraha.txt",
            "max_rows": 50_000,
        },
    ]

    print("\nDownloading Indian mythology / folktale datasets...")
    for c in candidates:
        out = raw_dir / c["out"]
        if out.exists():
            print(f"  Already exists: {out.name}")
            continue
        try:
            stream_to_file(
                c["id"], c.get("config"), c["split"], c["field"],
                out, max_rows=c.get("max_rows"),
            )
        except Exception as e:
            print(f"  Skipped {c['id']}: {e}")


# ---------------------------------------------------------------------------
# Merge & clean per language
# ---------------------------------------------------------------------------

_ODIA_RE  = re.compile(r"[\u0B00-\u0B7F]")
_HINDI_RE = re.compile(r"[\u0900-\u097F]")
_HTML_RE  = re.compile(r"<[^>]+>")
_URL_RE   = re.compile(r"https?://\S+")
_WS_RE    = re.compile(r"[ \t]+")


def _clean(line: str) -> str:
    line = _URL_RE.sub(" ", line)
    line = _HTML_RE.sub(" ", line)
    line = unicodedata.normalize("NFC", line)
    line = _WS_RE.sub(" ", line).strip()
    return line


def merge_and_clean(raw_files: list[Path], out_path: Path,
                    min_chars: int = 10) -> None:
    if out_path.exists():
        print(f"  Already exists: {out_path.name}")
        return
    seen: set[str] = set()
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for raw in raw_files:
            if not raw.exists():
                continue
            print(f"  Merging {raw.name}...")
            with open(raw, encoding="utf-8", errors="ignore") as f:
                for line in tqdm(f, desc=f"    {raw.stem}", unit="lines"):
                    for seg in re.split(r"(?<=[।॥\.\!\?])\s+", line):
                        cleaned = _clean(seg)
                        if len(cleaned) < min_chars or cleaned in seen:
                            continue
                        seen.add(cleaned)
                        out_f.write(cleaned + "\n")
                        written += 1
    print(f"  → {out_path.name}: {written:,} lines")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    raw_dir   = ROOT / "data" / "raw"
    clean_dir = ROOT / "data" / "clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # Hindi
    download_hindi_wikipedia(raw_dir)
    download_hindi_sangraha(raw_dir)

    # English
    download_english_wikipedia(raw_dir)
    download_english_c4(raw_dir)
    download_english_openwebtext(raw_dir)

    # Mythology / folktales
    download_mythology_datasets(raw_dir)

    # Merge into clean corpora
    print("\nMerging Hindi corpus...")
    merge_and_clean(
        [raw_dir / "wikipedia_hi.txt", raw_dir / "sangraha_verified_hi.txt",
         raw_dir / "mythology_hi_sangraha.txt"],
        clean_dir / "hindi_corpus.txt",
    )

    print("\nMerging English corpus...")
    merge_and_clean(
        [raw_dir / "wikipedia_en.txt", raw_dir / "c4_en.txt",
         raw_dir / "openwebtext_en.txt", raw_dir / "ramayana_en.txt",
         raw_dir / "mahabharata_en.txt", raw_dir / "indian_folktales_en.txt",
         raw_dir / "puranas_en.txt"],
        clean_dir / "english_corpus.txt",
    )

    print("\nAll multilingual data ready.")


if __name__ == "__main__":
    main()
