# Qwen3-TTS Easy Finetuning

This repository provides an easy-to-use workspace for fine-tuning the Qwen3-TTS model with custom speaker voices.

## Getting Started

### 1. Installation

**Using Docker**
```bash
docker build -t qwen3-tts-finetuner .
docker run --gpus all -it -v $(pwd):/workspace qwen3-tts-finetuner
```

**Using Python Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Flash Attention matching your CUDA/Torch version if needed
pip install flash-attn==2.8.3 --no-build-isolation
```

### 2. Data Preparation
Create a directory named `data` and put your fine-tuning data in `data/tts_train.jsonl`.
The jsonl format should look like this:
```json
{"audio": "/path/to/wav/1.wav", "text": "Transcript of the audio.", "ref_audio": "/path/to/wav/1.wav"}
```

### 3. Training
Run the training script by providing the path to your raw jsonl file and the speaker name.
```bash
chmod +x train.sh
./train.sh data/tts_train.jsonl <speaker_name>
```

The script will automatically:
1. Generate audio codes for your dataset and save it as `*_with_codes.jsonl`.
2. Start Tensorboard background process on port 6006.
3. Finetune the TTS model and save checkpoints to `output/`.

### 4. Testing
Once the fine-tuning is completed, you can use `quicktest.py` to synthesize new speech:
```bash
python quicktest.py \
    --model_path output/checkpoint-epoch-0 \
    --speaker <speaker_name> \
    --text "Hello, I am testing the new voice model!" \
    --output output.wav
```
