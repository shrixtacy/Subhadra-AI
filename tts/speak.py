"""
Odia TTS — uses facebook/mms-tts-ory for Odia speech synthesis.

  OdiaTTS.speak(text, output_path)  -> saves .wav file
  OdiaTTS.speak_stream(text)        -> returns WAV bytes for streaming
"""

from __future__ import annotations
import io
import sys
import yaml
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


class OdiaTTS:
    """MMS VITS wrapper for Odia text-to-speech synthesis."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        from transformers import VitsModel, VitsTokenizer

        cfg = _load_config()
        if model_dir is None:
            model_dir = ROOT / cfg["tts"]["output_dir"]
        model_dir = Path(model_dir)

        if model_dir.exists() and (model_dir / "config.json").exists():
            model_id = str(model_dir)
            print(f"Loading TTS from {model_dir}")
        else:
            model_id = "facebook/mms-tts-ory"
            print(f"Loading base TTS: {model_id}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = cfg["tts"]["sample_rate"]
        self.tokenizer = VitsTokenizer.from_pretrained(model_id)
        self.model = VitsModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        print(f"OdiaTTS ready  [{self.device}]")

    def speak(self, text: str, output_path: str | Path) -> None:
        import soundfile as sf
        wav, sr = self._synthesize(text)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), wav, sr)
        print(f"Audio saved -> {output_path}")

    def speak_stream(self, text: str) -> bytes:
        import soundfile as sf
        wav, sr = self._synthesize(text)
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()

    def _synthesize(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        wav = output.waveform[0].cpu().numpy().astype(np.float32)
        sr = self.model.config.sampling_rate
        return wav, sr


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python speak.py <odia_text> <output.wav>")
        sys.exit(1)

    tts = OdiaTTS()
    tts.speak(sys.argv[1], sys.argv[2])
