"""
Multilingual tokenizer wrapper — supports Odia (or), Hindi (hi), English (en).

Falls back to the existing odia_spm.model if the multilingual model hasn't been
trained yet, so the rest of the system keeps working during a phased rollout.

Language detection uses Unicode block heuristics (no extra deps required):
  - Odia   U+0B00–U+0B7F
  - Hindi  U+0900–U+097F  (Devanagari)
  - English  ASCII-dominant
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import List

import sentencepiece as spm

# ---------------------------------------------------------------------------
# Unicode heuristics
# ---------------------------------------------------------------------------

_ODIA_RE    = re.compile(r"[\u0B00-\u0B7F]")
_HINDI_RE   = re.compile(r"[\u0900-\u097F]")
_LATIN_RE   = re.compile(r"[A-Za-z]")

LANG_ODIA    = "or"
LANG_HINDI   = "hi"
LANG_ENGLISH = "en"


def detect_language(text: str) -> str:
    """Heuristic language detection based on Unicode character counts."""
    odia  = len(_ODIA_RE.findall(text))
    hindi = len(_HINDI_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    total = odia + hindi + latin or 1
    if odia / total >= 0.3:
        return LANG_ODIA
    if hindi / total >= 0.3:
        return LANG_HINDI
    return LANG_ENGLISH


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class MultilingualTokenizer:
    """
    Thin wrapper around a SentencePiece model that adds language-tag tokens
    and auto-detects language when not specified.

    Special tokens (must be in the trained vocab):
      <pad> <unk> <bos> <eos> <sep>
      <lang:or> <lang:hi> <lang:en>
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    SEP_TOKEN = "<sep>"

    LANG_TOKENS = {
        LANG_ODIA:    "<lang:or>",
        LANG_HINDI:   "<lang:hi>",
        LANG_ENGLISH: "<lang:en>",
    }

    def __init__(self, model_path: str | Path | None = None) -> None:
        root = Path(__file__).parent
        # Prefer multilingual model; fall back to Odia-only
        if model_path is None:
            multi = root / "multilingual_spm.model"
            model_path = multi if multi.exists() else root / "odia_spm.model"

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SentencePiece model not found: {self.model_path}\n"
                "Run: python tokenizer/train_multilingual_tokenizer.py"
            )

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(self.model_path))

        self.pad_id = self.sp.PieceToId(self.PAD_TOKEN)
        self.unk_id = self.sp.PieceToId(self.UNK_TOKEN)
        self.bos_id = self.sp.PieceToId(self.BOS_TOKEN)
        self.eos_id = self.sp.PieceToId(self.EOS_TOKEN)
        self.sep_id = self.sp.PieceToId(self.SEP_TOKEN)

        # Language tag IDs (may be <unk> if using old Odia-only model)
        self._lang_ids: dict[str, int] = {
            lang: self.sp.PieceToId(tok)
            for lang, tok in self.LANG_TOKENS.items()
        }

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        lang: str | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text, optionally prepending a language tag token."""
        ids: List[int] = self.sp.EncodeAsIds(text)
        if lang is None:
            lang = detect_language(text)
        lang_id = self._lang_ids.get(lang, self.unk_id)
        prefix = ([self.bos_id] if add_bos else [])
        # Only inject lang tag if it's a real token (not unk)
        if lang_id != self.unk_id:
            prefix.append(lang_id)
        suffix = ([self.eos_id] if add_eos else [])
        return prefix + ids + suffix

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        special = {self.pad_id, self.bos_id, self.eos_id, self.sep_id}
        special.update(self._lang_ids.values())
        if skip_special:
            ids = [i for i in ids if i not in special]
        return self.sp.DecodeIds(ids)

    def encode_chat(
        self,
        question: str,
        answer: str = "",
        lang: str | None = None,
    ) -> List[int]:
        """
        SFT chat format with language tag:
        <bos><lang:XX>Human: {question}<sep>Subhadra: {answer}<eos>
        """
        if lang is None:
            lang = detect_language(question)
        lang_id = self._lang_ids.get(lang, self.unk_id)

        prompt = f"Human: {question}"
        ids = [self.bos_id]
        if lang_id != self.unk_id:
            ids.append(lang_id)
        ids += self.sp.EncodeAsIds(prompt) + [self.sep_id]
        if answer:
            ids += self.sp.EncodeAsIds(f"Subhadra: {answer}") + [self.eos_id]
        return ids

    def __repr__(self) -> str:
        return (
            f"MultilingualTokenizer(vocab_size={self.vocab_size}, "
            f"model={self.model_path.name})"
        )


# ---------------------------------------------------------------------------
# Backward-compat alias so existing code that imports OdiaTokenizer still works
# ---------------------------------------------------------------------------
OdiaTokenizer = MultilingualTokenizer


if __name__ == "__main__":
    tok = MultilingualTokenizer()
    print(tok)
    for sample in [
        "ନମସ୍କାର, ଆପଣ କେମିତି ଅଛନ୍ତି?",
        "नमस्ते, आप कैसे हैं?",
        "Hello, how are you?",
    ]:
        lang = detect_language(sample)
        ids  = tok.encode(sample, add_bos=True, add_eos=True)
        print(f"[{lang}] {sample}")
        print(f"  IDs   : {ids[:10]}...")
        print(f"  Decode: {tok.decode(ids)}\n")
