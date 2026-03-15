"""
Subhadra terminal chat loop.
  - Text mode: type Odia text, get Odia reply
  - Voice mode: type 'voice' to switch, mic input + audio output
  - Ctrl+C to exit
"""

from __future__ import annotations
import sys
import yaml
import argparse
import tempfile
import os
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizer.odia_tokenizer import OdiaTokenizer
from model.subhadra import SubhadraConfig, SubhadraForCausalLM


def load_config() -> dict:
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)


def load_slm(cfg: dict, device: torch.device) -> tuple[SubhadraForCausalLM, OdiaTokenizer]:
    tok       = OdiaTokenizer()
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


def generate(model: SubhadraForCausalLM, tok: OdiaTokenizer,
             text: str, device: torch.device,
             max_new_tokens: int = 200, temperature: float = 0.8) -> str:
    ids       = tok.encode_chat(text)
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    out       = model.generate(input_ids, max_new_tokens=max_new_tokens,
                               temperature=temperature, eos_id=tok.eos_id)
    return tok.decode(out[0, len(ids):].tolist())


def record_audio(duration: int = 5, sample_rate: int = 16000) -> np.ndarray:
    """Record from microphone using sounddevice."""
    import sounddevice as sd
    print(f"  [ରେକର୍ଡ କରୁଛି {duration} ସେକେଣ୍ଡ...]")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def play_audio(wav_bytes: bytes, sample_rate: int = 22050) -> None:
    """Play WAV bytes through speakers."""
    import sounddevice as sd
    import soundfile as sf
    import io
    data, sr = sf.read(io.BytesIO(wav_bytes))
    sd.play(data, sr)
    sd.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Subhadra Chat CLI")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens",  type=int,   default=200)
    args = parser.parse_args()

    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print("  ସୁଭଦ୍ରା — ଓଡ଼ିଆ AI ସହାୟକ")
    print(f"  Device: {device}")
    print("  'voice' ଟାଇପ କରନ୍ତୁ ଭଏସ ମୋଡ ପାଇଁ")
    print("  Ctrl+C ଦ୍ୱାରା ବାହାର ହୁଅନ୍ତୁ")
    print(f"{'='*50}\n")

    model, tok = load_slm(cfg, device)

    # Lazy-load ASR/TTS only when voice mode is activated
    asr = None
    tts = None
    voice_mode = False

    try:
        while True:
            try:
                prompt = input("ଆପଣ: ").strip()
            except EOFError:
                break

            if not prompt:
                continue

            if prompt.lower() == "voice":
                if asr is None:
                    print("ଭଏସ ମୋଡ ଲୋଡ ହେଉଛି...")
                    from stt.transcribe import OdiaASR
                    from tts.speak import OdiaTTS
                    asr = OdiaASR()
                    tts = OdiaTTS()
                voice_mode = not voice_mode
                print(f"ଭଏସ ମୋଡ: {'ଚାଲୁ' if voice_mode else 'ବନ୍ଦ'}")
                continue

            if voice_mode and asr is not None:
                # Record mic input
                audio_array = record_audio(duration=5)
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio_array, 16000)
                    tmp_path = tmp.name
                prompt = asr.transcribe(tmp_path)
                os.unlink(tmp_path)
                print(f"ଆପଣ (ଭଏସ): {prompt}")

            reply = generate(model, tok, prompt, devi
ce,
                          max_new_tokens=args.max_tokens,
                          temperature=args.temperature)
            print(f"ସୁଭଦ୍ରା: {reply}\n")

            if voice_mode and tts is not None:
                wav_bytes = tts.speak_stream(reply)
                play_audio(wav_bytes, sample_rate=cfg["tts"]["sample_rate"])

    except KeyboardInterrupt:
        print("\nଧନ୍ୟବାଦ! ପୁଣି ଦେଖା ହେବ।")


if __name__ == "__main__":
    main()
