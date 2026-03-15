"""
Train a SentencePiece BPE tokenizer on the cleaned Odia corpus.
Saves:
  tokenizer/odia_spm.model
  tokenizer/odia_spm.vocab
"""

import os
import yaml
import sentencepiece as spm
from pathlib import Path

def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def train_tokenizer(
    input_file: str,
    model_prefix: str,
    vocab_size: int,
    character_coverage: float,
    special_tokens: list[str],
) -> None:
    """Train SentencePiece BPE model."""
    # Build the user_defined_symbols string (skip <unk> — it's built-in)
    user_symbols = [t for t in special_tokens if t != "<unk>"]

    print(f"Training SentencePiece BPE tokenizer...")
    print(f"  Input : {input_file}")
    print(f"  Prefix: {model_prefix}")
    print(f"  Vocab : {vocab_size}")

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        user_defined_symbols=",".join(user_symbols),
        # Treat whitespace as part of tokens (better for Indic scripts)
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
        normalization_rule_name="nfkc",
        input_sentence_size=5_000_000,   # cap for memory
        shuffle_input_sentence=True,
    )
    print(f"\nTokenizer saved:")
    print(f"  {model_prefix}.model")
    print(f"  {model_prefix}.vocab")

def main() -> None:
    cfg = load_config()
    tok_cfg = cfg["tokenizer"]
    data_cfg = cfg["data"]

    clean_corpus = Path(data_cfg["clean_dir"]) / "odia_corpus.txt"
    if not clean_corpus.exists():
        raise FileNotFoundError(
            f"Clean corpus not found at {clean_corpus}.\n"
            "Run: python data/clean_data.py first."
        )

    # model_prefix is relative to project root
    model_prefix = str(Path(__file__).parent.parent / tok_cfg["model_prefix"])
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)

    train_tokenizer(
        input_file=str(clean_corpus),
        model_prefix=model_prefix,
        vocab_size=tok_cfg["vocab_size"],
        character_coverage=tok_cfg["character_coverage"],
        special_tokens=tok_cfg["special_tokens"],
    )

if __name__ == "__main__":
    main()
