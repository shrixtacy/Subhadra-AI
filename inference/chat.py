"""
Subhadra multilingual terminal chat loop.
  - Supports Odia (or), Hindi (hi), English (en)
  - Auto-detects language from input and switches TTS voice accordingly
  - Text mode: type in any supported language, get a reply
  - Voice mode: type 'voice' to toggle mic input + audio output
  - 'lang or|hi|en' to force a language
  - Ctrl+C to exit
"""

from __future__ import annotations
import io
import sys
import yaml
import argparse
import tempfile
import os
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizer.multilingual_tokenizer import MultilingualTokenizer, detect_language
from model.subhadra import SubhadraConfig, SubhadraForCausalLM


def load_config() -> dict:
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)


def load_slm(cfg: dict, device: torch.device) -> tuple[SubhadraForCausalLM, MultilingualTokenizer]:
    tok       = MultilingualTokenizer()
    model_cfg = SubhadraConfig(**cfg["model"])
    model     = SubhadraForCausalLM(model_cfg).to(device)

    sft_dir = Path(cfg["sft"]["checkpoint_dir"])
    ckpts   = sorted(sft_dir.glob("sft_epoch*.pt")) if sft_dir.exists() else []
    if not ckpts:
        pt_dir = Path(cfg["pretrain"]["checkpoint_dir"])
        ckpts  = sorted(pt_dir.glob("ckpt_step*.pt")) if pt_dir.exists() else []
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded: {ckpts[-1].name}")
    else:
        print("WARNING: No checkpoint found. Using random weights.")
    model.eval()
    return model, tok


def generate(
    model: SubhadraForCausalLM,
    tok: MultilingualTokenizer,
    text: str,
    device: torch.device,
    lang: str | None = None,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
) -> str:
    ids       = tok.encode_chat(text, lang=lang)
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    out       = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_id=tok.eos_id,
    )
    return tok.decode(out[0, len(ids):].tolist())


def record_audio(duration: int = 5, sample_rate: int = 16000) -> np.ndarray:
    import sounddevice as sd
    print(f"  [Recording {duration}s — speak now...]")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def play_audio(wav_bytes: bytes) -> None:
    import sounddevice as sd
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(wav_bytes))
    sd.play(data, sr)
    sd.wait()


_LANG_LABELS = {"or": "Odia", "hi": "Hindi", "en": "English"}

_BANNER = """
╔══════════════════════════════════════════════════════╗
║   Subhadra — Multilingual AI (Odia · Hindi · English) ║
║   Indian Mythology & Folktales Edition               ║
╠══════════════════════════════════════════════════════╣
║  Commands:                                           ║
║    voice          — toggle voice mode                ║
║    lang or|hi|en  — force language                   ║
║    Ctrl+C         — exit                             ║
╚══════════════════════════════════════════════════════╝
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Subhadra Multilingual Chat CLI")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens",  type=int,   default=200)
    parser.add_argument("--lang",        type=str,   default=None,
                        choices=["or", "hi", "en"],
                        help="Force language (default: auto-detect)")
    args = parser.parse_args()

    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(_BANNER)
    print(f"  Device: {device}")

    model, tok = load_slm(cfg, device)

    # Lazy-load ASR/TTS only when voice mode is activated
    asr        = None
    tts        = None
    voice_mode = False
    forced_lang: str | None = args.lang

    try:
        while True:
            lang_label = _LANG_LABELS.get(forced_lang, "auto") if forced_lang else "auto"
            try:
                prompt = input(f"You [{lang_label}]: ").strip()
            except EOFError:
                break

            if not prompt:
                continue

            # ── Commands ──────────────────────────────────────────────────
            if prompt.lower() == "voice":
                if asr is None:
                    print("Loading voice modules...")
                    from stt.transcribe import OdiaASR
                    from tts.speak import MultilingualTTS
                    asr = OdiaASR()
                    tts = MultilingualTTS()
                voice_mode = not voice_mode
                print(f"Voice mode: {'ON' if voice_mode else 'OFF'}")
                continue

            if prompt.lower().startswith("lang "):
                parts = prompt.split()
                if len(parts) == 2 and parts[1] in ("or", "hi", "en"):
                    forced_lang = parts[1]
                    print(f"Language forced to: {_LANG_LABELS[forced_lang]}")
                else:
                    print("Usage: lang or|hi|en")
                continue

            # ── Voice input ───────────────────────────────────────────────
            if voice_mode and asr is not None:
                audio_array = record_audio(duration=5)
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio_array, 16000)
                    tmp_path = tmp.name
                prompt = asr.transcribe(tmp_path)
                os.unlink(tmp_path)
                print(f"You (voice): {prompt}")

            # ── Language detection ────────────────────────────────────────
            lang = forced_lang or detect_language(prompt)

            # ── Generate reply ────────────────────────────────────────────
            reply = generate(
                model, tok, prompt, device,
                lang=lang,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(f"Subhadra [{_LANG_LABELS.get(lang, lang)}]: {reply}\n")

            # ── Voice output (switches voice per detected language) ────────
            if voice_mode and tts is not None:
                wav_bytes = tts.speak_stream(reply, lang=lang)
                play_audio(wav_bytes)

    except KeyboardInterrupt:
        print("\nThank you! ধন্যবাদ! ଧନ୍ୟବାଦ!")


if __name__ == "__main__":
    main()
