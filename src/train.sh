#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH=${TOKENIZER_MODEL_PATH:-"Qwen/Qwen3-TTS-Tokenizer-12Hz"}
INIT_MODEL_PATH=${INIT_MODEL_PATH:-"Qwen/Qwen3-TTS-12Hz-0.6B-Base"}

# Inputs
RAW_JSONL=${1:-"data/tts_train.jsonl"}
SPEAKER_NAME=${2:-"my_speaker"}

if [ ! -f "${RAW_JSONL}" ]; then
  echo "Error: Input JSONL file not found at ${RAW_JSONL}"
  echo "Usage: ./train.sh <path_to_raw_jsonl> <speaker_name>"
  exit 1
fi

TRAIN_JSONL="${RAW_JSONL%.jsonl}_with_codes.jsonl"
OUTPUT_DIR="output/"

BATCH_SIZE=2
LR=1e-7
EPOCHS=2

# 1. Prepare Data
if [ -f "${TRAIN_JSONL}" ]; then
  echo "=> Data already prepared, skipping prepare_data.py. Using: ${TRAIN_JSONL}"
else
  echo "=> Preparing data to generate audio codes..."
  python prepare_data.py \
    --device ${DEVICE} \
    --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
    --input_jsonl ${RAW_JSONL} \
    --output_jsonl ${TRAIN_JSONL}
fi

# 2. Run Tensorboard
if ! pgrep -f "tensorboard --logdir logs" > /dev/null; then
    echo "=> Starting Tensorboard on port 6006..."
    tensorboard --logdir logs --bind_all --port 6006 &
fi

# 3. Fine-tuning
echo "=> Starting training for speaker: ${SPEAKER_NAME}..."
python sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME} \
  --resume_from_checkpoint latest

echo "=> Training completed!"
