"""
Multilingual TTS with language-aware female voices.

Voice mapping (female voices):
  Odia    (or) — facebook/mms-tts-ory          (MMS VITS)
  Hindi   (hi) — facebook/mms-tts-hin          (MMS VITS)
  English (en) — tts_models/en/ljspeech/vits   (Coqui TTS, female LJSpeech)
                 fallback: gTTS (Google TTS, female)

Usage:
  tts = MultilingualTTS()
  tts.speak("Hello, how are you?", "out.wav", lang="en")
  wav_bytes = tts.speak_stream("नमस्ते", lang="hi")

  # Auto-detect language:
  wav_bytes = tts.speak_stream("ନମସ୍କାର")
"""

from __future__ import annotations
import io
import sys
import yaml
from pathlib import Path
from typing import Literal

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.multilingual_tokenizer import detect_language, LANG_ODIA, LANG_HINDI, LANG_ENGLISH


def _load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Per-language MMS VITS voice (Odia & Hindi — female where available)
# ---------------------------------------------------------------------------

# MMS model IDs — these are the HuggingFace facebook/mms-tts-* models.
# For Odia and Hindi, MMS provides a single speaker voice.
_MMS_MODELS = {
    LANG_ODIA:  "facebook/mms-tts-ory",
    LANG_HINDI: "facebook/mms-tts-hin",
}


class _MMSVoice:
    """Wrapper for facebook/mms-tts-* VITS models."""

    def __init__(self, model_id: str, device: torch.device) -> None:
        from transformers import VitsModel, VitsTokenizer
        print(f"  Loading MMS TTS: {model_id}")
        self.tokenizer = VitsTokenizer.from_pretrained(model_id)
        self.model     = VitsModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device      = device
        self.sample_rate = self.model.config.sampling_rate

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        wav = out.waveform[0].cpu().numpy().astype(np.float32)
        return wav, self.sample_rate


# ---------------------------------------------------------------------------
# English female voice — Coqui TTS (LJSpeech VITS) with gTTS fallback
# ---------------------------------------------------------------------------

class _EnglishFemaleVoice:
    """
    Uses Coqui TTS (tts_models/en/ljspeech/vits) for a female English voice.
    Falls back to gTTS (Google TTS) if Coqui is not installed.
    """

    COQUI_MODEL = "tts_models/en/ljspeech/vits"

    def __init__(self) -> None:
        self._backend: str = "none"
        self._tts = None
        self.sample_rate = 22050

        # Try Coqui TTS first
        try:
            from TTS.api import TTS as CoquiTTS
            print(f"  Loading Coqui TTS: {self.COQUI_MODEL}")
            self._tts = CoquiTTS(self.COQUI_MODEL, progress_bar=False, gpu=False)
            self._backend = "coqui"
            self.sample_rate = 22050
            print("  English female voice: Coqui TTS (LJSpeech VITS)")
        except Exception as e:
            print(f"  Coqui TTS unavailable ({e}), falling back to gTTS")

        # Fallback: gTTS
        if self._backend == "none":
            try:
                import gtts  # noqa: F401
                self._backend = "gtts"
                self.sample_rate = 24000
                print("  English female voice: gTTS (Google TTS)")
            except ImportError:
                print("  WARNING: No English TTS backend found. Install TTS or gtts.")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if self._backend == "coqui":
            return self._coqui_synth(text)
        if self._backend == "gtts":
            return self._gtts_synth(text)
        # Last resort: silence
        return np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate

    def _coqui_synth(self, text: str) -> tuple[np.ndarray, int]:
        import soundfile as sf
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self._tts.tts_to_file(text=text, file_path=tmp_path)
            wav, sr = sf.read(tmp_path, dtype="float32")
        finally:
            os.unlink(tmp_path)
        return wav, sr

    def _gtts_synth(self, text: str) -> tuple[np.ndarray, int]:
        from gtts import gTTS
        import soundfile as sf
        import tempfile, os
        tts = gTTS(text=text, lang="en", slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            tts.save(tmp_path)
            # Convert mp3 → numpy via librosa
            import librosa
            wav, sr = librosa.load(tmp_path, sr=self.sample_rate, mono=True)
        finally:
            os.unlink(tmp_path)
        return wav.astype(np.float32), sr


# ---------------------------------------------------------------------------
# Main multilingual TTS class
# ---------------------------------------------------------------------------

class MultilingualTTS:
    """
    Language-aware TTS that switches female voice automatically when the
    language changes.

    Voices:
      Odia    → facebook/mms-tts-ory  (MMS VITS)
      Hindi   → facebook/mms-tts-hin  (MMS VITS)
      English → Coqui LJSpeech VITS   (female) / gTTS fallback
    """

    def __init__(self) -> None:
        cfg = _load_config()
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = cfg["tts"]["sample_rate"]
        self._voices: dict[str, object] = {}
        print(f"MultilingualTTS initializing  [{self.device}]")
        self._load_all()

    def _load_all(self) -> None:
        for lang, model_id in _MMS_MODELS.items():
            try:
                self._voices[lang] = _MMSVoice(model_id, self.device)
            except Exception as e:
                print(f"  WARNING: Could not load {lang} voice: {e}")
        # English female voice
        try:
            self._voices[LANG_ENGLISH] = _EnglishFemaleVoice()
        except Exception as e:
            print(f"  WARNING: Could not load English voice: {e}")
        print("MultilingualTTS ready.")

    def _get_voice(self, lang: str):
        voice = self._voices.get(lang)
        if voice is None:
            # Fallback chain: hi → or → en
            for fallback in [LANG_HINDI, LANG_ODIA, LANG_ENGLISH]:
                voice = self._voices.get(fallback)
                if voice:
                    break
        return voice

    def speak(
        self,
        text: str,
        output_path: str | Path,
        lang: str | None = None,
    ) -> None:
        import soundfile as sf
        wav, sr = self._synthesize(text, lang)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), wav, sr)
        print(f"Audio saved → {output_path}  [{lang or detect_language(text)}]")

    def speak_stream(
        self,
        text: str,
        lang: str | None = None,
    ) -> bytes:
        import soundfile as sf
        wav, sr = self._synthesize(text, lang)
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()

    def _synthesize(self, text: str, lang: str | None) -> tuple[np.ndarray, int]:
        if lang is None:
            lang = detect_language(text)
        voice = self._get_voice(lang)
        if voice is None:
            print("WARNING: No TTS voice available — returning silence.")
            return np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate
        return voice.synthesize(text)


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------
OdiaTTS = MultilingualTTS


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python speak.py <text> <output.wav> [lang: or|hi|en]")
        sys.exit(1)
    text = sys.argv[1]
    out  = sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) > 3 else None
    tts  = MultilingualTTS()
    tts.speak(text, out, lang=lang)
