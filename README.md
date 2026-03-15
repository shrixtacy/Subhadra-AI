# Subhadra — Native Odia Small Language Model

A fully custom Odia AI with text chat and voice capabilities, built from scratch in Python + PyTorch.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Phase 1 — Tokenizer

**Step 1: Download corpora**
```bash
python data/download_data.py
```

**Step 2: Clean text**
```bash
python data/clean_data.py
```

**Step 3: Train BPE tokenizer (vocab=32000)**
```bash
python tokenizer/train_tokenizer.py
```

**Test tokenizer:**
```bash
python tokenizer/odia_tokenizer.py
```

---

## Phase 2 — Model Architecture

**Sanity check (forward pass + param count):**
```bash
python model/subhadra.py
```

Expected output: `SubhadraForCausalLM — ~85.0M parameters`

---

## Phase 3 — Pretraining

```bash
python train/pretrain.py
```

- Resumes automatically from latest checkpoint in `train/checkpoints/`
- Logs loss every 100 steps, saves every 1000 steps
- Stops when loss < 2.5

---

## Phase 4 — Chat Fine-Tuning (SFT)

**Build SFT dataset (500+ Odia Q&A pairs):**
```bash
python data/build_sft_data.py
```

**Fine-tune:**
```bash
python train/sft.py
```

---

## Phase 5 — Odia Speech-to-Text

**Fine-tune Whisper-small on Shrutilipi Odia:**
```bash
python stt/finetune_whisper.py
```

**Test transcription:**
```bash
python stt/transcribe.py path/to/audio.wav
```

---

## Phase 6 — Odia Text-to-Speech

**Train VITS on IndicTTS Odia:**
```bash
python tts/train_tts.py
```

**Test synthesis:**
```bash
python tts/speak.py "ନମସ୍କାର" output.wav
```

---

## Phase 7 — API Server

```bash
cd api
uvicorn server:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET  /health`       — model load status
- `POST /chat`         — `{"message": "...", "mode": "text"}`
- `POST /voice-chat`   — multipart audio upload → WAV response

---

## Phase 8 — Inference CLI

```bash
python inference/chat.py
```

- Type Odia text to chat
- Type `voice` to toggle voice mode (mic input + speaker output)
- `Ctrl+C` to exit

---

## Config

All hyperparameters live in `config.yaml` at the project root.

---

## Hardware Recommendations

| Phase        | Minimum         | Recommended      |
|--------------|-----------------|------------------|
| Tokenizer    | CPU             | CPU              |
| Pretraining  | 1× GPU 8GB VRAM | 1× A100 40GB     |
| SFT          | 1× GPU 8GB VRAM | 1× RTX 3090      |
| Whisper STT  | 1× GPU 8GB VRAM | 1× RTX 3090      |
| VITS TTS     | 1× GPU 8GB VRAM | 1× RTX 3090      |
| Inference    | CPU (slow)      | 1× GPU any VRAM  |
