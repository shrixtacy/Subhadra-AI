"""
Subhadra FastAPI server.
Endpoints:
  POST /chat        — text in, text out
  POST /voice-chat  — audio in, audio out
  GET  /health      — model load status
"""

from __future__ import annotations
import io
import sys
import asyncio
import yaml
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizer.odia_tokenizer import OdiaTokenizer
from model.subhadra import SubhadraConfig, SubhadraForCausalLM


def load_config() -> dict:
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

class ModelState:
    tokenizer: Optional[OdiaTokenizer]       = None
    slm:       Optional[SubhadraForCausalLM] = None
    asr:       Optional[object]              = None
    tts:       Optional[object]              = None
    device:    torch.device                  = torch.device("cpu")
    status:    dict                          = {"slm": False, "asr": False, "tts": False}

state = ModelState()


def _load_slm(cfg: dict) -> None:
    state.tokenizer = OdiaTokenizer()
    model_cfg = SubhadraConfig(**cfg["model"])
    state.slm = SubhadraForCausalLM(model_cfg).to(state.device)
    # Load best SFT checkpoint if available
    sft_dir = Path(cfg["sft"]["checkpoint_dir"])
    ckpts   = sorted(sft_dir.glob("sft_epoch*.pt")) if sft_dir.exists() else []
    if not ckpts:
        pt_dir = Path(cfg["pretrain"]["checkpoint_dir"])
        ckpts  = sorted(pt_dir.glob("ckpt_step*.pt")) if pt_dir.exists() else []
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location="cpu")
        state.slm.load_state_dict(ckpt["model_state"])
        print(f"SLM loaded from {ckpts[-1].name}")
    else:
        print("WARNING: No SLM checkpoint found. Using random weights.")
    state.slm.eval()
    state.status["slm"] = True


def _load_asr(cfg: dict) -> None:
    try:
        from stt.transcribe import OdiaASR
        state.asr = OdiaASR()
        state.status["asr"] = True
    except Exception as e:
        print(f"ASR load failed (non-fatal): {e}")


def _load_tts(cfg: dict) -> None:
    try:
        from tts.speak import OdiaTTS
        state.tts = OdiaTTS()
        state.status["tts"] = True
    except Exception as e:
        print(f"TTS load failed (non-fatal): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on {state.device}...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_slm, cfg)
    await loop.run_in_executor(None, _load_asr, cfg)
    await loop.run_in_executor(None, _load_tts, cfg)
    print(f"Model status: {state.status}")
    yield
    print("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Subhadra", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    mode:    str = "text"
    max_new_tokens: int = 200
    temperature:    float = 0.8
    top_k:          int   = 50
    top_p:          float = 0.9


class ChatResponse(BaseModel):
    reply: str
    mode:  str


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _generate_reply(message: str, max_new_tokens: int, temperature: float,
                    top_k: int, top_p: float) -> str:
    tok = state.tokenizer
    ids = tok.encode_chat(message)
    input_ids = torch.tensor([ids], dtype=torch.long).to(state.device)
    out = state.slm.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tok.eos_id,
    )
    new_ids = out[0, len(ids):].tolist()
    return tok.decode(new_ids)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "models": state.status, "device": str(state.device)}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not state.status["slm"]:
        raise HTTPException(503, "SLM not loaded")
    loop  = asyncio.get_event_loop()
    reply = await loop.run_in_executor(
        None, _generate_reply,
        req.message, req.max_new_tokens, req.temperature, req.top_k, req.top_p,
    )
    return ChatResponse(reply=reply, mode="text")


@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)) -> StreamingResponse:
    if not state.status["slm"]:
        raise HTTPException(503, "SLM not loaded")
    if not state.status["asr"]:
        raise HTTPException(503, "ASR model not loaded")
    if not state.status["tts"]:
        raise HTTPException(503, "TTS model not loaded")

    # Save upload to temp file
    import tempfile, os
    suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    loop = asyncio.get_event_loop()
    try:
        # STT
        odia_text = await loop.run_in_executor(None, state.asr.transcribe, tmp_path)
        # SLM
        reply_text = await loop.run_in_executor(None, _generate_reply, odia_text, 200, 0.8, 50, 0.9)
        # TTS
        wav_bytes  = await loop.run_in_executor(None, state.tts.speak_stream, reply_text)
    finally:
        os.unlink(tmp_path)

    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    uvicorn.run("server:app", host=cfg["api"]["host"], port=cfg["api"]["port"], reload=False)
