"""
Odia TTS — uses facebook/mms-tts-ory with pitch shifting for female voice.

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


def _pitch_shift(audio: np.ndarray, sr: int, semitones: float = 4.0) -> np.ndarray:
    """Shift pitch up by N semitones using resampling trick (no librosa needed)."""
    factor = 2 ** (semitones / 12.0)
    # Speed up by factor (raises pitch), then resample back to original length
    original_len = len(audio)
    indices = np.linspace(0, original_len - 1, int(original_len / factor))
    shifted = np.interp(indices, np.arange(original_len), audio)
    # Resample back to original length
    result_indices = np.linspace(0, len(shifted) - 1, original_len)
    return np.interp(result_indices, np.arange(len(shifted)), shifted).astype(np.float32)


class OdiaTTS:
    """MMS VITS Odia TTS with pitch shift for female-sounding voice."""

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
        # Shift pitch up 4 semitones for female-sounding voice
        wav = _pitch_shift(wav, sr, semitones=4.0)
        return wav, sr


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python speak.py <odia_text> <output.wav>")
        sys.exit(1)

    tts = OdiaTTS()
    tts.speak(sys.argv[1], sys.argv[2])
