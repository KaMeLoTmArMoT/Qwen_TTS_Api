# AGENTS.md

## Overview
`Qwen_TTS_Api` (`quen-tts`) is a lightweight FastAPI web service providing HTTP endpoints for speech synthesis using Qwen3-TTS (`CustomVoice`). It synthesizes audio spans (narration, dialogue, pause) into single concatenated WAV files.

## Tech Stack & Tooling
- **Python**: `>=3.11`
- **Package Management**: `uv` (`pyproject.toml`, `uv.lock`)
- **Linting & Formatting**: `ruff` (`uv run ruff check --fix`, `uv run ruff format`)
- **Framework**: `FastAPI`, `uvicorn`
- **Machine Learning**: `qwen-tts`, `torch`, `soundfile`, `huggingface_hub`

## Useful Commands
- **Environment Sync**: `uv sync`
- **Run Development Server**: `uv run uvicorn server.app:app --host 0.0.0.0 --port 8001`
- **Lint Code**: `uv run ruff check --fix`
- **Format Code**: `uv run ruff format`
- **Build Docker Image**: `docker build -t qwen-tts-api .`
