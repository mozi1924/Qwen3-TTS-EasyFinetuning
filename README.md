# Qwen3-TTS Easy Finetuning

This repository provides an easy-to-use workspace for fine-tuning the Qwen3-TTS model. It streamlines the process from raw audio data to a custom voice model with high stability and expressiveness.

---

### 🎙️ Why Fine-tuning instead of Zero-shot? / 为什么要进行微调？

While zero-shot voice cloning is convenient, fine-tuning (SFT) offers significant advantages for production-quality results:

- **Stability (音色更稳定)**: Fine-tuned models capture the nuances of the target speaker more accurately, resulting in consistent output across different sentences.
- **Natural Control (支持自然语言指导)**: It supports natural language tone/rhythm guidance (e.g., "Speak sadly", "Faster pace"), allowing more expressive speech synthesis.
- **Accent-free Cross-lingual (无母语口音)**: Prevents the "original language accent" during cross-lingual inference (e.g., a Chinese speaker's voice used for English speech will sound like a native English speaker).

---

## ✨ Features
- **Integrated Pipeline**: Audio splitting, ASR transcription, cleaning, and tokenization in one click.
- **Modern WebUI**: Premium Gradio interface for data preparation, training, and testing.
- **Robust CLI**: Complete command-line tools for automated workflows.
- **Training Presets**: Hardcoded optimized settings for different model sizes (0.6B / 1.7B).
- **Docker Support**: Pre-configured environment for easy setup.

---

## 🚀 Getting Started

### 1. Installation

**Using Docker (Recommended)**
```bash
docker compose up -d  # Using docker-compose
# OR
docker build -t qwen3-tts-finetuner .
docker run --gpus all -it -v $(pwd):/workspace qwen3-tts-finetuner
```

**Using Python Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Flash Attention matching your CUDA/Torch version
pip install flash-attn==2.8.3 --no-build-isolation
```

### 2. Using the WebUI (Easiest)
Run the Gradio WebUI to manage the entire process through your browser:
```bash
python src/webui.py
```
- **Data Prep Tab**: Upload raw audio -> Split -> ASR -> Tokenize.
- **Training Tab**: Select dataset -> Configure settings -> Start Tensorboard -> Train.
- **Inference Tab**: Load your trained checkpoint and generate audio!

### 3. Using the CLI (Professional)
The `src/cli.py` provides a unified entry point for all operations:

**Step A: Prepare Data**
Put your raw `.wav` files in a folder (e.g., `raw-dataset/my_speaker/`).
```bash
python src/cli.py prepare --input_dir raw-dataset/my_speaker --speaker_name my_speaker
```

**Step B: Start Training**
```bash
python src/cli.py train --experiment_name exp1 --speaker_name my_speaker --epochs 3
```

**Step C: Run Inference**
```bash
python src/cli.py infer --checkpoint output/exp1/checkpoint-epoch-2 --speaker my_speaker --text "Hello world! This is my custom voice."
```

---

## 📂 Project Structure
- `src/webui.py`: Main Gradio interface.
- `src/cli.py`: Unified command-line interface.
- `src/sft_12hz.py`: Core fine-tuning logic.
- `src/step1_audio_split.py`: Audio preprocessing & segmentation.
- `src/step2_asr_clean.py`: Automatic transcription & data labeling.
- `src/prepare_data.py`: Pre-tokenizing audio into discrete codes (Step 3).

---

## 🤝 Acknowledgments
- Based on [Qwen3-TTS](https://github.com/qwenLM/Qwen3-tts) and [Qwen3-ASR](https://github.com/qwenLM/Qwen3-asr).
- Training presets inspired by community contributions (e.g., [rekuenkdr](https://github.com/rekuenkdr)).

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mozi1924/Qwen3-TTS-EasyFinetuning&type=date&legend=top-left)](https://www.star-history.com/#mozi1924/Qwen3-TTS-EasyFinetuning&type=date&legend=top-left)