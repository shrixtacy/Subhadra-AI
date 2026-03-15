"""
Odia tokenizer wrapper around the trained SentencePiece model.
Provides encode() and decode() with special token support.
"""

from __future__ import annotations
import sentencepiece as spm
from pathlib import Path
from typing import List

class OdiaTokenizer:
    PAD_TOKEN  = "<pad>"
    UNK_TOKEN  = "<unk>"
    BOS_TOKEN  = "<bos>"
    EOS_TOKEN  = "<eos>"
    SEP_TOKEN  = "<sep>"

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).parent / "odia_spm.model"
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SentencePiece model not found: {self.model_path}\n"
                "Run: python tokenizer/train_tokenizer.py first."
            )
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(self.model_path))

        # Cache special token IDs
        self.pad_id  = self.sp.PieceToId(self.PAD_TOKEN)
        self.unk_id  = self.sp.PieceToId(self.UNK_TOKEN)
        self.bos_id  = self.sp.PieceToId(self.BOS_TOKEN)
        self.eos_id  = self.sp.PieceToId(self.EOS_TOKEN)
        self.sep_id  = self.sp.PieceToId(self.SEP_TOKEN)

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text to token IDs."""
        ids: List[int] = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text."""
        special = {self.pad_id, self.bos_id, self.eos_id, self.sep_id}
        if skip_special:
            ids = [i for i in ids if i not in special]
        return self.sp.DecodeIds(ids)

    def encode_chat(self, question: str, answer: str = "") -> List[int]:
        """
        Encode a chat turn in SFT format:
        <bos>Human: {question}<sep>Subhadra: {answer}<eos>
        """
        prompt = f"Human: {question}"
        ids = [self.bos_id] + self.sp.EncodeAsIds(prompt) + [self.sep_id]
        if answer:
            ids += self.sp.EncodeAsIds(f"Subhadra: {answer}") + [self.eos_id]
        return ids

    def __repr__(self) -> str:
        return f"OdiaTokenizer(vocab_size={self.vocab_size}, model={self.model_path.name})"


if __name__ == "__main__":
    import sys
    tok = OdiaTokenizer()
    print(tok)
    test = "ନମସ୍କାର, ଆପଣ କେମିତି ଅଛନ୍ତି?"
    ids = tok.encode(test, add_bos=True, add_eos=True)
    print(f"Input : {test}")
    print(f"IDs   : {ids}")
    print(f"Decode: {tok.decode(ids)}")
