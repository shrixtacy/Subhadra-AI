# Subhadra — Native Odia AI

<p align="center">
  <img src="https://img.shields.io/badge/language-Odia-orange" />
  <img src="https://img.shields.io/badge/model-50M%20params-blue" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/badge/python-3.10%2B-yellow" />
  <img src="https://img.shields.io/github/stars/shrixtacy/Subhadra-AI?style=social" />
</p>

A fully custom Odia language AI built from scratch — custom tokenizer, pretrained language model, fine-tuned speech recognition, and text-to-speech. Designed to bring native Odia language understanding to AI.

---

## What is Subhadra?

Subhadra is an end-to-end Odia AI system with:

- **Custom BPE Tokenizer** — 32,000 vocab trained on Odia corpora
- **Subhadra LM** — 50M parameter transformer pretrained on Odia text
- **SFT** — Fine-tuned on Odia Q&A instruction pairs
- **STT** — Whisper-small fine-tuned on AI4Bharat Shrutilipi Odia dataset
- **TTS** — Facebook MMS Odia voice synthesis
- **REST API** — FastAPI server with `/chat` and `/voice-chat` endpoints

---

## Project Structure

```
Subhadra-AI/
├── config.yaml              # All hyperparameters
├── requirements.txt
│
├── data/
│   ├── download_data.py     # Download raw Odia corpora
│   ├── clean_data.py        # Clean and merge corpora
│   ├── build_sft_data.py    # Generate SFT instruction dataset
│   ├── odia_sft.jsonl       # SFT Q&A pairs
│   ├── raw/                 # Raw corpus files
│   └── clean/               # Cleaned corpus
│
├── tokenizer/
│   ├── train_tokenizer.py   # Train BPE tokenizer
│   ├── odia_tokenizer.py    # Tokenizer wrapper
│   ├── odia_spm.model       # Trained SentencePiece model
│   └── odia_spm.vocab       # Vocabulary
│
├── model/
│   └── subhadra.py          # Transformer architecture
│
├── train/
│   ├── pretrain.py          # Pretraining loop
│   ├── sft.py               # Supervised fine-tuning
│   └── checkpoints/         # Saved checkpoints (gitignored)
│
├── stt/
│   ├── finetune_whisper.py  # Fine-tune Whisper for Odia ASR
│   └── transcribe.py        # Transcription inference
│
├── tts/
│   ├── train_tts.py         # TTS model setup
│   └── speak.py             # Speech synthesis inference
│
├── inference/
│   └── chat.py              # Terminal chat CLI
│
└── api/
    └── server.py            # FastAPI REST server
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/shrixtacy/Subhadra-AI.git
cd Subhadra-AI
pip install -r requirements.txt
```

### 2. Download checkpoints

Download the pretrained checkpoints and extract them:

```
train/checkpoints/       ← pretrain_ckpts.zip
train/sft_checkpoints/   ← sft_ckpts.zip
stt/whisper_odia/        ← whisper_ckpts.zip
tts/vits_odia/           ← tts_ckpts.zip
```

### 3. Run the API

```bash
python api/server.py
```

Server starts at `http://localhost:8000`

### 4. Chat via CLI

```bash
python inference/chat.py
```

---

## API Reference

### `GET /health`
Returns model load status.

```json
{
  "status": "ok",
  "models": {"slm": true, "asr": true, "tts": true},
  "device": "cuda"
}
```

### `POST /chat`
Text chat with Subhadra.

**Request:**
```json
{
  "message": "ଓଡ଼ିଶାର ରାଜଧାନୀ କ'ଣ?",
  "max_new_tokens": 200,
  "temperature": 0.8
}
```

**Response:**
```json
{
  "reply": "ଓଡ଼ିଶାର ରାଜଧାନୀ ହେଉଛି ଭୁବନେଶ୍ୱର।",
  "mode": "text"
}
```

### `POST /voice-chat`
Send a `.wav` audio file, get a `.wav` audio response.

```bash
curl -X POST http://localhost:8000/voice-chat \
  -F "audio=@input.wav" \
  --output reply.wav
```

---

## Training from Scratch

### Phase 1 — Tokenizer

```bash
python data/download_data.py
python data/clean_data.py
python tokenizer/train_tokenizer.py
```

### Phase 2 — Pretrain

```bash
python train/pretrain.py
```

Resumes automatically from latest checkpoint. Stops when loss < 2.5.

### Phase 3 — SFT

```bash
python data/build_sft_data.py
python train/sft.py
```

### Phase 4 — STT (Whisper fine-tuning)

Requires HuggingFace token for `ai4bharat/shrutilipi` (gated dataset):

```bash
huggingface-cli login
python stt/finetune_whisper.py
```

### Phase 5 — TTS

```bash
python tts/train_tts.py
```

---

## Hardware Requirements

| Phase       | Minimum GPU VRAM | Recommended       |
|-------------|------------------|-------------------|
| Tokenizer   | CPU only         | CPU only          |
| Pretrain    | 8GB              | 16GB+ (P100/A100) |
| SFT         | 8GB              | 16GB+             |
| Whisper STT | 8GB              | 16GB+             |
| TTS         | 4GB              | 8GB+              |
| Inference   | CPU (slow)       | Any GPU           |

Training was done on Kaggle/Google Colab with free P100/T4 GPUs.

---

## Configuration

All hyperparameters are in `config.yaml`:

```yaml
model:
  n_layers: 8
  d_model: 512
  n_heads: 8
  d_ff: 2048
  max_seq_len: 1024
  vocab_size: 32000

pretrain:
  batch_size: 16
  lr: 3e-4
  max_steps: 100000
  target_loss: 2.5
```

---

## Roadmap

- [ ] Female Odia TTS voice (custom recorded dataset)
- [ ] Larger model (200M+ params)
- [ ] Expanded SFT dataset (10k+ pairs)
- [ ] HuggingFace Spaces demo
- [ ] Mobile app integration
- [ ] Odia Wikipedia Q&A fine-tuning

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [AI4Bharat](https://ai4bharat.org) — Shrutilipi Odia ASR dataset
- [Facebook MMS](https://huggingface.co/facebook/mms-tts-ory) — Odia TTS base model
- [OpenAI Whisper](https://github.com/openai/whisper) — ASR base model
- [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) — Odia pretraining corpus
