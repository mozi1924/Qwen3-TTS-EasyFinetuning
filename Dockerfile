FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV USE_HF=1
ENV VLLM_USE_MODELSCOPE=true

RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    ffmpeg \
    sox \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir qwen-tts==0.1.1 qwen-asr==0.0.6 --no-deps

# Install Flash Attention
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl || pip install flash-attn==2.8.3 --no-build-isolation

COPY src/ /workspace/src/

EXPOSE 7860 6006

ENV PYTHONPATH="/workspace/src"

CMD ["python", "src/webui.py"]
