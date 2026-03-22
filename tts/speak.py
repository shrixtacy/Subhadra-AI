"""
Odia TTS — uses ai4bharat/indic-parler-tts for female Odia voice.

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
    """Indic Parler-TTS wrapper for female Odia speech synthesis."""

    VOICE_PROMPT = (
        "A female speaker with a clear, pleasant voice delivers the text "
        "at a moderate pace in Odia language."
    )

    def __init__(self, model_dir: str | Path | None = None) -> None:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        cfg = _load_config()
        self.sample_rate = cfg["tts"]["sample_rate"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_id = "ai4bharat/indic-parler-tts"
        print(f"Loading Indic Parler-TTS from {model_id}...")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.desc_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )
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
        desc_inputs = self.desc_tokenizer(
            self.VOICE_PROMPT, return_tensors="pt"
        ).to(self.device)
        text_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            gen = self.model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=text_inputs.input_ids,
                prompt_attention_mask=text_inputs.attention_mask,
            )
        wav = gen.cpu().numpy().squeeze().astype(np.float32)
        sr = self.model.config.sampling_rate
        return wav, sr


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python speak.py <odia_text> <output.wav>")
        sys.exit(1)

    tts = OdiaTTS()
    tts.speak(sys.argv[1], sys.argv[2])
