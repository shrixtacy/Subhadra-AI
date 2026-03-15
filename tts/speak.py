"""
Odia TTS — load trained VITS model and synthesize speech.

  OdiaTTS.speak(text, output_path)  → saves .wav file
  OdiaTTS.speak_stream(text)        → returns WAV bytes for streaming
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


def _find_checkpoint(model_dir: Path) -> tuple[Path, Path]:
    """
    Locate best_model*.pth + config.json inside Coqui's output tree.
    Coqui saves to: <output_dir>/vits_odia-<date>/<run>/
    """
    # Prefer best_model, fall back to latest checkpoint
    for pattern in ("**/best_model*.pth", "**/*.pth"):
        ckpts = sorted(model_dir.rglob(pattern.replace("**/", "")))
        if not ckpts:
            ckpts = sorted(model_dir.rglob(pattern.split("/")[-1]))
        if ckpts:
            ckpt = ckpts[-1]
            cfg  = ckpt.parent / "config.json"
            if cfg.exists():
                return ckpt, cfg

    raise FileNotFoundError(
        f"No VITS checkpoint + config.json found under {model_dir}\n"
        "Run: python tts/train_tts.py first."
    )


class OdiaTTS:
    """Coqui VITS wrapper for Odia text-to-speech synthesis."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        from TTS.api import TTS as CoquiTTS

        cfg = _load_config()
        if model_dir is None:
            model_dir = ROOT / cfg["tts"]["output_dir"]
        model_dir = Path(model_dir)

        if not model_dir.exists():
            raise FileNotFoundError(
                f"TTS model directory not found: {model_dir}\n"
                "Run: python tts/train_tts.py first."
            )

        ckpt_path, config_path = _find_checkpoint(model_dir)
        print(f"Loading VITS from {ckpt_path.name}")

        self.device      = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = cfg["tts"]["sample_rate"]
        self.tts         = CoquiTTS(
            model_path=str(ckpt_path),
            config_path=str(config_path),
            progress_bar=False,
        )
        self.tts.to(self.device)
        print(f"OdiaTTS ready  [{self.device}]  sr={self.sample_rate}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str, output_path: str | Path) -> None:
        """Synthesize Odia text and save as a .wav file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.tts.tts_to_file(text=text, file_path=str(output_path))
        print(f"Audio saved → {output_path}")

    def speak_stream(self, text: str) -> bytes:
        """
        Synthesize Odia text and return raw WAV bytes.
        Suitable for streaming over HTTP or playing in-memory.
        """
        import soundfile as sf

        wav_list: list = self.tts.tts(text=text)
        wav = np.array(wav_list, dtype=np.float32)

        buf = io.BytesIO()
        sf.write(buf, wav, self.sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python speak.py <odia_text> <output.wav>")
        sys.exit(1)

    text        = sys.argv[1]
    output_path = sys.argv[2]

    tts = OdiaTTS()
    tts.speak(text, output_path)
