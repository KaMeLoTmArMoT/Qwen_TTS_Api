# app.py
import io
import os
import threading

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from qwen_tts import Qwen3TTSModel

from server.chapter_render import Span, render_chapter_customvoice
from server.pydantic_models import SUPPORTED_MODELS, ChapterReq, GenBatchReq, GenReq, LoadReq

app = FastAPI()
_lock = threading.Lock()
print("Docs: http://127.0.0.1:8001/docs")
LOCAL_CUSTOMVOICE_17B = os.getenv("QWEN_TTS_MODEL_DIR", "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")


class State:
    model: Qwen3TTSModel | None = None
    model_id: str | None = None
    device: str = "cuda:0"
    dtype: str = "float16"
    attn_implementation: str = "flash_attention_2"


STATE = State()


def _dtype_from_str(s: str):
    s = s.lower()
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError("dtype must be float16 or bfloat16")


@app.post("/load")
def load(req: LoadReq):
    print("load req:", req)
    if req.model_id not in SUPPORTED_MODELS:
        raise HTTPException(400, f"Unsupported model_id. Choose one of: {SUPPORTED_MODELS}")

    with _lock:
        # reload if different model requested
        if STATE.model is not None and STATE.model_id == req.model_id and STATE.dtype == req.dtype:
            return {"ok": True, "model_id": STATE.model_id, "dtype": STATE.dtype}

        # unload old
        STATE.model = None
        torch.cuda.empty_cache()

        STATE.model_id = req.model_id

        STATE.dtype = req.dtype
        STATE.attn_implementation = req.attn_implementation

        model_source = req.model_id
        if req.model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice":
            model_source = LOCAL_CUSTOMVOICE_17B  # local dir => no download

        STATE.model = Qwen3TTSModel.from_pretrained(
            model_source,
            device_map=STATE.device,
            dtype=_dtype_from_str(req.dtype),
            attn_implementation=req.attn_implementation,
        )

        return {
            "ok": True,
            "model_id": STATE.model_id,
            "dtype": STATE.dtype,
            "attn_implementation": STATE.attn_implementation,
            "languages": STATE.model.get_supported_languages(),
            "speakers": STATE.model.get_supported_speakers(),
        }


@app.post("/unload")
def unload():
    print("Unloading model")
    with _lock:
        STATE.model = None
        STATE.model_id = None
        torch.cuda.empty_cache()
        return {"ok": True}


@app.get("/state")
def state():
    return {
        "loaded": STATE.model is not None,
        "model_id": STATE.model_id,
        "device": STATE.device,
        "dtype": STATE.dtype,
        "attn_implementation": STATE.attn_implementation,
    }


def _ensure_loaded():
    if STATE.model is None:
        raise HTTPException(409, "Model not loaded. Call /load first.")


@app.post("/generate")
def generate(req: GenReq):
    print("generate req:", req)
    with _lock:
        _ensure_loaded()
        wavs, sr = STATE.model.generate_custom_voice(
            text=req.text,
            language=req.language,
            speaker=req.speaker,
            instruct=req.instruct,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
        )

    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.post("/generate_batch")
def generate_batch(req: GenBatchReq):
    print("generate_batch req:", req)
    # Expand scalar language/speaker/instruct to lists (so it matches README-style batch usage)
    n = len(req.text)
    language = req.language if isinstance(req.language, list) else [req.language] * n
    speaker = req.speaker if isinstance(req.speaker, list) else [req.speaker] * n
    instruct = req.instruct if isinstance(req.instruct, list) else [req.instruct] * n

    if not (len(language) == len(speaker) == len(instruct) == n):
        raise HTTPException(400, "text/language/speaker/instruct must have same length")

    with _lock:
        _ensure_loaded()
        wavs, sr = STATE.model.generate_custom_voice(
            text=req.text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
        )

    # simplest: return concatenated zip is nicer, but keep it minimal:
    # Here: return first WAV if you want; for real batch, return a zip (next iteration).
    # For now, return WAVs as a single multi-channel WAV is usually wrong.
    # -> better: base64 list or ZIP. We'll do base64 list.
    import base64

    outs = []
    for w in wavs:
        b = io.BytesIO()
        sf.write(b, w, sr, format="WAV")
        outs.append(base64.b64encode(b.getvalue()).decode("ascii"))

    return {"sample_rate": sr, "wavs_b64": outs}


@app.post("/generate_chapter")
def generate_chapter(req: ChapterReq):
    print("generate_chapter req:", req)
    with _lock:
        _ensure_loaded()

        spans = [Span(kind=s.kind, text=s.text, pause_ms=s.pause_ms) for s in req.spans]
        wav_bytes, sr = render_chapter_customvoice(
            model=STATE.model,
            spans=spans,
            language=req.language,
            speaker_map=req.speaker_map,
            instruct_map=req.instruct_map,
            lead_in_ms=req.lead_in_ms,
            gap_ms=req.gap_ms,
            default_pause_ms=req.default_pause_ms,
            fade_ms=req.fade_ms,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
        )

    filename = f"{req.book_id or 'book'}_{req.chapter_id or 'chapter'}.wav"
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
