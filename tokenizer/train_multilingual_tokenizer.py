"""
Train a multilingual SentencePiece BPE tokenizer on Odia + Hindi + English corpora.

Input files (all plain-text, one sentence per line):
  data/clean/odia_corpus.txt
  data/clean/hindi_corpus.txt
  data/clean/english_corpus.txt

Output:
  tokenizer/multilingual_spm.model
  tokenizer/multilingual_spm.vocab

Run: python tokenizer/train_multilingual_tokenizer.py
"""

from __future__ import annotations
import yaml
import tempfile
import sentencepiece as spm
from pathlib import Path

# Cap per language — keeps temp file small and training stable
# Hindi corpus is 5.6 GB so we sample it, Odia/English are smaller
MAX_LINES: dict[str, int] = {
    "odia_corpus.txt":    500_000,
    "hindi_corpus.txt":   500_000,
    "english_corpus.txt": 1_000_000,
}
DEFAULT_CAP = 500_000


def load_config() -> dict:
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)


def train(
    input_files: list[str],
    model_prefix: str,
    vocab_size: int,
    character_coverage: float,
    special_tokens: list[str],
) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     encoding="utf-8", delete=False) as tmp:
        for path in input_files:
            p = Path(path)
            if not p.exists():
                print(f"  WARNING: {p} not found — skipping")
                continue
            cap = MAX_LINES.get(p.name, DEFAULT_CAP)
            print(f"  Adding {p.name}  (cap={cap:,} lines)")
            written = 0
            with open(p, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        tmp.write(line + "\n")
                        written += 1
                        if written >= cap:
                            break
            print(f"    → {written:,} lines written")
        tmp_path = tmp.name

    tmp_size_mb = Path(tmp_path).stat().st_size / 1024 / 1024
    print(f"\nTemp corpus: {tmp_size_mb:.1f} MB")

    user_symbols = [t for t in special_tokens if t != "<unk>"]

    print(f"Training multilingual BPE tokenizer (vocab={vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="bpe",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>",
        bos_piece="<bos>", eos_piece="<eos>",
        user_defined_symbols=",".join(user_symbols),
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
        normalization_rule_name="nfkc",
        input_sentence_size=1_500_000,   # spm internal cap as safety net
        shuffle_input_sentence=True,
    )
    Path(tmp_path).unlink(missing_ok=True)
    print(f"Saved: {model_prefix}.model / .vocab")


def main() -> None:
    cfg = load_config()
    tok_cfg = cfg["tokenizer"]

    input_files = [
        "data/clean/odia_corpus.txt",
        "data/clean/hindi_corpus.txt",
        "data/clean/english_corpus.txt",
    ]

    extra = ["<sep>", "<lang:or>", "<lang:hi>", "<lang:en>"]
    special_tokens = tok_cfg.get("special_tokens", []) + extra
    seen: set[str] = set()
    special_tokens = [t for t in special_tokens if not (t in seen or seen.add(t))]

    train(
        input_files=input_files,
        model_prefix="tokenizer/multilingual_spm",
        vocab_size=tok_cfg.get("multilingual_vocab_size", 48000),
        character_coverage=tok_cfg.get("character_coverage", 0.9999),
        special_tokens=special_tokens,
    )


if __name__ == "__main__":
    main()
