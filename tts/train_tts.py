"""
Train Coqui VITS TTS on AI4Bharat IndicTTS Odia single-speaker dataset.
  - Architecture : VITS (end-to-end, no mel intermediate)
  - Dataset      : ai4bharat/IndicTTS, Odia subset
  - Sample rate  : 22050 Hz
  - Output       : tts/vits_odia/

Install deps first:
  pip install TTS==0.22.0 trainer soundfile
"""

from __future__ import annotations
import sys
import csv
import yaml
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_config(cfg_path: str | Path | None = None) -> dict:
    if cfg_path is None:
        cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset download — IndicTTS Odia → LJSpeech-style layout
# ---------------------------------------------------------------------------

def download_indictts_odia(data_dir: Path, sample_rate: int = 22050) -> tuple[Path, Path]:
    """
    Download AI4Bharat IndicTTS Odia and convert to LJSpeech layout:
      data_dir/
        wavs/          ← resampled .wav files
        metadata.csv   ← stem|text|text  (pipe-delimited, no header)
    Returns (wavs_dir, metadata_csv).
    """
    import soundfile as sf
    from datasets import load_dataset, Audio

    wavs_dir      = data_dir / "wavs"
    metadata_path = data_dir / "metadata.csv"

    if metadata_path.exists() and any(wavs_dir.glob("*.wav")):
        print(f"IndicTTS data already present at {data_dir}")
        return wavs_dir, metadata_path

    wavs_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading AI4Bharat IndicTTS Odia dataset...")

    # Try known dataset IDs in order
    ds = None
    candidates = [
        ("ai4bharat/IndicTTS", "or"),
        ("ai4bharat/indic-tts-odia", None),
        ("ai4bharat/IndicTTS-Odia", None),
    ]
    for ds_id, cfg_name in candidates:
        try:
            kwargs = dict(split="train", trust_remote_code=True)
            if cfg_name:
                kwargs["name"] = cfg_name
            ds = load_dataset(ds_id, **kwargs)
            print(f"Loaded dataset: {ds_id}" + (f" (config={cfg_name})" if cfg_name else ""))
            break
        except Exception as e:
            print(f"  {ds_id}: {e}")

    if ds is None:
        raise RuntimeError(
            "Could not load IndicTTS Odia dataset from HuggingFace.\n"
            "Check https://huggingface.co/ai4bharat for the current dataset ID."
        )

    # Resample to target sample rate
    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))

    # Detect text column
    text_col = next(
        (c for c in ("text", "sentence", "transcription") if c in ds.column_names),
        None,
    )
    if text_col is None:
        raise ValueError(f"No text column found. Columns: {ds.column_names}")

    print(f"Writing {len(ds)} samples (text_col='{text_col}')...")
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        for i, row in enumerate(ds):
            stem     = f"odia_{i:05d}"
            wav_path = wavs_dir / f"{stem}.wav"
            audio    = row["audio"]
            sf.write(str(wav_path), audio["array"], audio["sampling_rate"])
            text = row[text_col].strip()
            writer.writerow([stem, text, text])   # stem|normalized|raw

    print(f"Dataset ready: {len(ds)} samples → {data_dir}")
    return wavs_dir, metadata_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(cfg_path: str | Path | None = None) -> None:
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.vits import Vits, VitsAudioConfig
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor
    from TTS.config import BaseDatasetConfig
    from TTS.tts.configs.shared_configs import CharactersConfig
    from trainer import Trainer, TrainerArgs

    cfg     = load_config(cfg_path)
    tts_cfg = cfg["tts"]

    output_dir = ROOT / tts_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = ROOT / "data" / "tts_odia"
    wavs_dir, metadata_path = download_indictts_odia(data_dir, tts_cfg["sample_rate"])

    # --- audio config ---
    audio_config = VitsAudioConfig(
        sample_rate=tts_cfg["sample_rate"],
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    )

    # --- Odia character set ---
    odia_chars = (
        "ଁଂଃଅଆଇଈଉଊଋଌଏଐଓଔ"
        "କଖଗଘଙଚଛଜଝଞଟଠଡଢଣ"
        "ତଥଦଧନପଫବଭମଯରଲଳଵ"
        "ଶଷସହ଼ଽାିୀୁୂୃୄେୈୋୌ୍ୖୗ"
        " "
    )
    characters_config = CharactersConfig(
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=odia_chars,
        punctuations="!\"',-./:;? ",
        phonemes=None,
    )

    # --- dataset config ---
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train=str(metadata_path),
        path=str(data_dir),
        language="or",
    )

    # --- VITS config ---
    config = VitsConfig(
        audio=audio_config,
        run_name="vits_odia",
        batch_size=tts_cfg["batch_size"],
        eval_batch_size=4,
        num_loader_workers=2,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=tts_cfg["epochs"],
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        phoneme_language=None,
        phoneme_cache_path=None,
        compute_input_seq_cache=True,
        print_step=50,
        print_eval=True,
        output_path=str(output_dir),
        datasets=[dataset_config],
        characters=characters_config,
        save_step=1000,
        save_n_checkpoints=3,
        save_best_after=1000,
    )

    ap                    = AudioProcessor.init_from_config(config)
    tokenizer, config     = TTSTokenizer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    model = Vits(config, ap, tokenizer, speaker_manager=None)

    trainer_args = TrainerArgs(
        mixed_precision=True,
    )

    trainer = Trainer(
        trainer_args,
        config,
        output_path=str(output_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print(f"Starting VITS training — {tts_cfg['epochs']} epochs → {output_dir}")
    trainer.fit()
    print(f"TTS model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Odia VITS TTS")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
