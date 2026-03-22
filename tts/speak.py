"""
Odia TTS — load fine-tuned MMS VITS model and synthesize speech.

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
    """HuggingFace MMS VITS wrapper for Odia text-to-speech synthesis."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        from transformers import VitsModel, VitsTokenizer

        cfg = _load_config()
        if model_dir is None:
            model_dir = ROOT / cfg["tts"]["output_dir"]
        model_dir = Path(model_dir)

        # Fall back to base MMS model if fine-tuned model not available
        if model_dir.exists() and (model_dir / "config.json").exists():
            model_id = str(model_dir)
            print(f"Loading fine-tuned TTS from {model_dir}")
        else:
            model_id = "facebook/mms-tts-ory"
            print(f"Fine-tuned model not found, using base: {model_id}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = cfg["tts"]["sample_rate"]
        self.tokenizer = VitsTokenizer.from_pretrained(model_id)
        self.model = VitsModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        print(f"OdiaTTS ready  [{self.device}]  sr={self.sample_rate}")

    def speak(self, text: str, output_path: str | Path) -> None:
        """Synthesize Odia text and save as a .wav file."""
        import soundfile as sf

        wav = self._synthesize(text)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), wav, self.sample_rate)
        print(f"Audio saved -> {output_path}")

    def speak_stream(self, text: str) -> bytes:
        """Synthesize Odia text and return raw WAV bytes for streaming."""
        import soundfile as sf

        wav = self._synthesize(text)
        buf = io.BytesIO()
        sf.write(buf, wav, self.sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()

    def _synthesize(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        return output.waveform[0].cpu().numpy()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python speak.py <odia_text> <output.wav>")
        sys.exit(1)

    tts = OdiaTTS()
    tts.speak(sys.argv[1], sys.argv[2])
