FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
WORKDIR /app

RUN pip install -U pip
RUN pip install fastapi uvicorn[standard] soundfile qwen-tts "huggingface_hub[cli]"

RUN pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3%2Bcu126torch2.6-cp311-cp311-linux_x86_64.whl"

RUN huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir /models/Qwen3-TTS-12Hz-1.7B-CustomVoice

COPY server/ /app/server/
ENV QWEN_TTS_MODEL_DIR=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
EXPOSE 8001

CMD ["uvicorn", "server.app:app", "--host=0.0.0.0", "--port=8001"]
