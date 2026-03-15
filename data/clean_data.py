"""
Clean raw Odia text corpus:
  - Strip HTML tags
  - Normalize Unicode to NFC
  - Keep only Odia script + ASCII punctuation/digits + Devanagari dandas
  - Drop lines with < 10 Odia characters
  - Deduplicate lines
  - Write merged clean corpus → data/clean/odia_corpus.txt

Run: python data/clean_data.py
"""

from __future__ import annotations
import re
import unicodedata
from pathlib import Path
from typing import Iterator
import yaml
from tqdm import tqdm


# Odia Unicode block U+0B00–U+0B7F
# Keep: Odia, ASCII printable, Devanagari danda/double-danda (।॥), newline
_STRIP_RE  = re.compile(r"[^\u0B00-\u0B7F\u0020-\u007E\u0964\u0965\n]")
_HTML_RE   = re.compile(r"<[^>]+>")
_URL_RE    = re.compile(r"https?://\S+")
_MULTI_WS  = re.compile(r"[ \t]+")
_MIN_ODIA  = re.compile(r"[\u0B00-\u0B7F]")


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def clean_line(line: str) -> str:
    line = _URL_RE.sub(" ", line)
    line = _HTML_RE.sub(" ", line)
    line = unicodedata.normalize("NFC", line)
    line = _STRIP_RE.sub("", line)
    line = _MULTI_WS.sub(" ", line).strip()
    return line


def iter_raw_files(raw_dir: Path) -> Iterator[Path]:
    for p in sorted(raw_dir.glob("*.txt")):
        yield p


def main() -> None:
    cfg       = load_config()
    raw_dir   = Path(cfg["data"]["raw_dir"])
    clean_dir = Path(cfg["data"]["clean_dir"])
    clean_dir.mkdir(parents=True, exist_ok=True)

    out_path = clean_dir / "odia_corpus.txt"
    seen: set[str] = set()
    total_written  = 0
    total_skipped  = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for raw_file in iter_raw_files(raw_dir):
            print(f"\nCleaning {raw_file.name} ...")
            file_written = 0
            with open(raw_file, encoding="utf-8", errors="ignore") as in_f:
                for raw_line in tqdm(in_f, desc=f"  {raw_file.stem}", unit="lines"):
                    # Each line may be a full document — split on sentence-ending dandas
                    for segment in re.split(r"(?<=[।॥\.\!\?])\s+", raw_line):
                        cleaned = clean_line(segment)
                        if len(_MIN_ODIA.findall(cleaned)) < 10:
                            total_skipped += 1
                            continue
                        if cleaned in seen:
                            total_skipped += 1
                            continue
                        seen.add(cleaned)
                        out_f.write(cleaned + "\n")
                        total_written += 1
                        file_written  += 1
            print(f"  → {file_written:,} lines kept")

    print(f"\nClean corpus: {total_written:,} lines kept, "
          f"{total_skipped:,} skipped → {out_path}")
    print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
