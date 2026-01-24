# pydantic_models.py
from pydantic import BaseModel, Field
from typing import List, Union, Literal, Dict

SUPPORTED_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
]


class LoadReq(BaseModel):
    model_id: str = Field(default=SUPPORTED_MODELS[0])
    dtype: str = "float16"
    attn_implementation: str = "flash_attention_2"


class GenReq(BaseModel):
    text: str
    language: str = "Auto"
    speaker: str = "Ryan"
    instruct: str = ""
    max_new_tokens: int = 1024
    do_sample: bool = False


class GenBatchReq(BaseModel):
    text: List[str]
    language: Union[str, List[str]] = "Auto"
    speaker: Union[str, List[str]] = "Ryan"
    instruct: Union[str, List[str]] = ""
    max_new_tokens: int = 1024
    do_sample: bool = False


class SpanIn(BaseModel):
    kind: Literal["narr", "dialog", "pause"]
    text: str = ""
    pause_ms: int = 0


class ChapterReq(BaseModel):
    book_id: str | None = None
    chapter_id: str | None = None

    spans: list[SpanIn]

    language: str = "English"
    speaker_map: Dict[str, str] = {"narr": "Aiden", "dialog": "Ryan"}
    instruct_map: Dict[str, str] = {
        "narr": "Professional audiobook narration; steady pace, clear consonants.",
        "dialog": "Natural dialogue; slightly more expressive than narration.",
    }

    lead_in_ms: int = 600
    gap_ms: int = 60
    default_pause_ms: int = 450
    fade_ms: int = 18

    max_new_tokens: int = 1024
    do_sample: bool = False
