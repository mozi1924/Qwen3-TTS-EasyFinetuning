# Qwen3-TTS Easy Finetuning

<p align="center">
  <a href="./README.md">English</a> | <b>简体中文</b>
</p>

这是一个专为 **Qwen3-TTS** 模型设计的易用微调工作站。它简化了从原始音频数据到构建高质量、高稳定性、极具表现力的自定义语音模型的全流程。

---

### 📚 教程文章

您可以阅读我的详细图文教程以获取完整指南：

👉 [中文教程](https://mozi1924.com/article/qwen3-tts-finetuning-zh/) | [English Tutorial](https://mozi1924.com/article/qwen3-tts-finetuning-en/)

### 🎙️ 为什么选择微调而非零样本 (Zero-shot)？

虽然零样本语音克隆非常便捷，但针对生产级的应用，微调 (SFT) 具有显著优势：

- **更高的音色稳定性**: 微调模型能更精确地捕捉目标人物的音质特征，确保不同文本下的输出高度一致。
- **卓越的口吻控制**: 支持通过自然语言进行情绪和节奏转换引导（如“悲伤地说话”、“语速加快”），让合成语音更具表现力。
- **无母语口音干扰**: 彻底解决跨语言合成时的“原本语种口音”问题（例如：用纯正中文音色合成英文时，听起来会像母语级英语使用者，而非带有中式口音）。

---

## ✨ 核心特性

- **一站式流水线**: 集成音频切分、ASR 转录、多轮清洗及 Tokenization，一键完成数据准备。
- **现代化 WebUI**: 基于 Gradio 设计的高级界面，涵盖数据准备、训练监控及推理测试。
- **强大 CLI 工具**: 提供完整的命令行接口，便于自动化脚本集成。
- **针对性预设**: 针对 0.6B 和 1.7B 不同规模的模型，内置了经过优化的训练参数。
- **完善的 Docker 支持**: 预配置环境镜像，实现即插即用。

---

## 🚀 快速上手

### 1. 安装环境

**使用 Docker (推荐)**
```bash
# 默认从 GHCR 拉取预构建镜像
docker compose up -d

# 如果需要强制本地构建
docker compose up -d --build
```

**使用 Python 虚拟环境**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 安装与您的 CUDA/Torch 版本匹配的 Flash Attention
pip install flash-attn==2.8.3 --no-build-isolation
```

### 2. 使用 WebUI (最简便)
直接启动 Gradio WebUI，在浏览器中管理整个流程：
```bash
python src/webui.py
```
- **数据准备面板**: 上传音频 -> 切分 -> ASR 识别 -> 转换为编码。
- **模型训练面板**: 选择数据集 -> 配置参数 -> 开启 Tensorboard -> 开始训练。
- **语音推理面板**: 加载训练好的 Checkpoint，立即生成您的专属语音！

### 3. 使用 CLI (进阶/专业)
`src/cli.py` 提供了一个统一的操作入口：

**步骤 A: 准备数据**
将原始 `.wav` 文件放入文件夹（如 `raw-dataset/my_speaker/`）。
```bash
python src/cli.py prepare --input_dir raw-dataset/my_speaker --speaker_name my_speaker
```

**步骤 B: 开始训练**
```bash
python src/cli.py train --experiment_name exp1 --speaker_name my_speaker --epochs 3
```

**步骤 C: 执行推理**
```bash
python src/cli.py infer --checkpoint output/exp1/checkpoint-epoch-2 --speaker my_speaker --text "你好，世界！这是我微调的自定义音色。"
```

---

## 📂 项目结构

- `src/webui.py`: 主 Gradio 交互界面。
- `src/cli.py`: 统一命令行入口。
- `src/sft_12hz.py`: 核心微调逻辑（监督微调）。
- `src/step1_audio_split.py`: 音频预处理与分段。
- `src/step2_asr_clean.py`: 自动化语音转文字及数据清洗。
- `src/prepare_data.py`: 将音频预处理为离散编码 (Step 3)。

---

## 🤝 致谢

- 基于 [Qwen3-TTS](https://github.com/qwenLM/Qwen3-tts) 和 [Qwen3-ASR](https://github.com/qwenLM/Qwen3-asr)。
- 训练预设逻辑参考了社区贡献者（如 [rekuenkdr](https://github.com/rekuenkdr)）。

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mozi1924/Qwen3-TTS-EasyFinetuning&type=date&legend=top-left)](https://www.star-history.com/#mozi1924/Qwen3-TTS-EasyFinetuning&type=date&legend=top-left)
