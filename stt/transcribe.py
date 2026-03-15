"""
Odia ASR — load fine-tuned Whisper and transcribe audio files.

  OdiaASR.transcribe(audio_path)        -> Odia text  (.wav / .mp3)
  OdiaASR.transcribe_array(array, sr)   -> Odia text  (raw numpy array)

Long audio (>30 s) is automatically chunked into 25-second windows and
the transcriptions are joined in order.
"""

from __future__ import annotations
import sys
import yaml
from pathlib import Path
from typing import Union

import torch
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
WHISPER_SR     = 16_000          # Whisper always expects 16 kHz
CHUNK_SEC      = 25              # chunk length — safely under Whisper's 30 s window
CHUNK_SAMPLES  = CHUNK_SEC * WHISPER_SR


def _load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _load_audio(path: str | Path) -> np.ndarray:
    """
    Load any audio file to a mono float32 numpy array at 16 kHz.
    Tries soundfile (fast, .wav/.flac) first, falls back to librosa (.mp3 etc.).
    """
    path = Path(path)
    ext  = path.suffix.lower()

    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported audio format '{ext}'. Supported: {SUPPORTED_EXTS}")
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # soundfile handles wav/flac without ffmpeg
    if ext in {".wav", ".flac"}:
        try:
            import soundfile as sf
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if audio.ndim == 2:                  # stereo → mono
                audio = audio.mean(axis=1)
            if sr != WHISPER_SR:
                import scipy.signal as sps
                samples = int(len(audio) * WHISPER_SR / sr)
                audio   = sps.resample(audio, samples).astype(np.float32)
            return audio
        except ImportError:
            pass   # fall through to librosa

    # librosa handles mp3 / ogg / m4a (requires ffmpeg for mp3)
    try:
        import librosa
        audio, _ = librosa.load(str(path), sr=WHISPER_SR, mono=True)
        return audio.astype(np.float32)
    except ImportError:
        raise ImportError(
            "Install audio deps: pip install soundfile scipy librosa"
        )


class OdiaASR:
    """Fine-tuned Whisper wrapper for Odia speech recognition."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        cfg = _load_config()
        if model_dir is None:
            model_dir = ROOT / cfg["stt"]["output_dir"]
        model_dir = Path(model_dir)

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Fine-tuned Whisper model not found at {model_dir}\n"
                "Run: python stt/finetune_whisper.py"
            )

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(str(model_dir))
        self.model     = WhisperForConditionalGeneration.from_pretrained(
            str(model_dir)
        ).to(self.device)
        self.model.eval()
        print(f"OdiaASR loaded from {model_dir.name}  [{self.device}]")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str | Path) -> str:
        """
        Transcribe a .wav or .mp3 file to Odia text.
        Automatically chunks audio longer than 25 seconds.
        """
        audio = _load_audio(audio_path)
        return self.transcribe_array(audio, sr=WHISPER_SR)

    def transcribe_array(self, audio: np.ndarray, sr: int = WHISPER_SR) -> str:
        """
        Transcribe a raw numpy float32 audio array to Odia text.
        Resamples to 16 kHz if needed. Chunks long audio automatically.
        """
        # Resample if caller passed a different sample rate
        if sr != WHISPER_SR:
            import scipy.signal as sps
            samples = int(len(audio) * WHISPER_SR / sr)
            audio   = sps.resample(audio, samples).astype(np.float32)

        # Split into chunks and transcribe each
        chunks = [
            audio[i : i + CHUNK_SAMPLES]
            for i in range(0, len(audio), CHUNK_SAMPLES)
        ]
        parts = [self._run_whisper(chunk) for chunk in chunks]
        return " ".join(p for p in parts if p.strip())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_whisper(self, audio_chunk: np.ndarray) -> str:
        """Run Whisper on a single ≤30 s chunk."""
        features = self.processor(
            audio_chunk,
            sampling_rate=WHISPER_SR,
            return_tensors="pt",
        ).input_features.to(self.device)

        with torch.inference_mode():
            ids = self.model.generate(
                features,
                language="or",
                task="transcribe",
                num_beams=5,
            )
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file.wav|.mp3>")
        sys.exit(1)

    asr  = OdiaASR()
    path = sys.argv[1]
    print(f"Transcribing: {path}")
    text = asr.transcribe(path)
    print(f"\nTranscription:\n{text}")
