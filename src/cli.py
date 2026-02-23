#!/usr/bin/env python3
"""
Qwen3-TTS Easy Finetuning — CLI Main Program

A command-line interface that mirrors the WebUI functionality.
Supports data preparation, training, and inference via subcommands.

Usage:
    python cli.py prepare --input_dir /path/to/wavs --speaker_name my_speaker
    python cli.py train   --experiment_name exp1 --speaker_name my_speaker
    python cli.py infer   --checkpoint output/exp1/checkpoint-epoch-1 --text "Hello world"
"""

import os
import sys
import time
from utils import get_model_path, get_project_root, resolve_path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────── Utilities ──────────────────────────

# Removed local get_model_path as it's now imported from utils


def print_header(text):
    """Print a styled section header."""
    width = 60
    print()
    print("═" * width)
    print(f"  {text}")
    print("═" * width)


def print_step(text):
    """Print a step indicator."""
    print(f"\n  ▶ {text}")


def consume_generator(gen):
    """Consume a generator, printing progress messages to the console."""
    for item in gen:
        if isinstance(item, dict):
            msg_type = item.get("type", "")
            if msg_type == "progress":
                pct = item.get("progress", 0)
                desc = item.get("desc", "")
                bar_len = 30
                filled = int(bar_len * pct)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r    [{bar}] {pct*100:5.1f}% {desc}", end="", flush=True)
            elif msg_type == "train_progress":
                epoch = item.get("epoch", 0)
                step = item.get("step", 0)
                loss = item.get("loss", 0.0)
                if isinstance(step, int):
                    print(f"\r    📈 Epoch {epoch} | Step {step:4d} | Loss: {loss:.6f}  ", end="", flush=True)
                else:
                    print(f"\n    📈 Epoch {epoch} | {step}")
            elif msg_type == "done":
                print(f"\n    ✅ {item.get('msg', 'Done!')}")
            elif msg_type == "error":
                print(f"\n    ❌ {item.get('msg', 'Unknown error')}")
            else:
                pass  # Ignore unknown types
        elif isinstance(item, str):
            print(f"    {item}")
    print()  # Final newline


# ─────────────────── Subcommands ────────────────────────

def cmd_prepare(args):
    """Run the full data preparation pipeline (Steps 1 → 2 → 3)."""
    print_header("🎙️ Qwen3-TTS Full Data Preparation")
    cmd_split(args)
    cmd_asr(args)
    cmd_tokenize(args)
    print_header("✅ Full Preparation Complete!")


def cmd_split(args):
    """Step 1: Audio Split & Resample."""
    print_step("Step 1: Audio Split & Resample")
    from step1_audio_split import run_step_1
    
    speaker_dir = resolve_path(os.path.join("final-dataset", args.speaker_name))
    audio_24k_dir = os.path.join(speaker_dir, "audio_24k")
    ref_audio = resolve_path(args.ref_audio) if args.ref_audio else None
    
    consume_generator(run_step_1(resolve_path(args.input_dir), audio_24k_dir, ref_audio))


def cmd_asr(args):
    """Step 2: ASR Transcription & Cleaning."""
    print_step("Step 2: ASR Transcription & Cleaning")
    from step2_asr_clean import run_step_2
    
    speaker_dir = resolve_path(os.path.join("final-dataset", args.speaker_name))
    audio_24k_dir = os.path.join(speaker_dir, "audio_24k")
    output_jsonl = os.path.join(speaker_dir, "tts_train.jsonl")
    
    use_hf = args.model_source == "HuggingFace"
    resolved_asr = get_model_path(args.asr_model, use_hf)
    
    ref_24k = os.path.join(audio_24k_dir, "ref_24k.wav")
    ref_path = ref_24k if os.path.exists(ref_24k) else ""
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu.replace("cuda:", "") if args.gpu != "cpu" else ""
    consume_generator(run_step_2(audio_24k_dir, ref_path, output_jsonl, resolved_asr, batch_size=args.batch_size))


def cmd_tokenize(args):
    """Step 3: Data Tokenization."""
    print_step("Step 3: Data Tokenization")
    from prepare_data import run_prepare
    
    speaker_dir = resolve_path(os.path.join("final-dataset", args.speaker_name))
    input_jsonl = os.path.join(speaker_dir, "tts_train.jsonl")
    
    # Save to logs/experiment_name/
    log_dir = resolve_path(os.path.join("logs", args.experiment_name))
    os.makedirs(log_dir, exist_ok=True)
    output_codes_jsonl = os.path.join(log_dir, "tts_train_with_codes.jsonl")
    
    resolved_tokenizer = get_model_path("Qwen/Qwen3-TTS-Tokenizer-12Hz", use_hf=False)
    device = "cuda:0" if args.gpu != "cpu" else "cpu"
    
    consume_generator(run_prepare(device, resolved_tokenizer, input_jsonl, output_codes_jsonl))


def cmd_train(args):
    """Run fine-tuning training."""
    import subprocess

    print_header("🏋️ Qwen3-TTS Fine-tuning")
    print(f"  Experiment   : {args.experiment_name}")
    print(f"  Speaker      : {args.speaker_name}")
    print(f"  Base Model   : {args.init_model}")
    print(f"  GPU Device   : {args.gpu}")
    print(f"  Batch Size   : {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Grad Accum   : {args.grad_acc}")

    train_jsonl = resolve_path(os.path.join("logs", args.experiment_name, "tts_train_with_codes.jsonl"))
    if not os.path.exists(train_jsonl):
        print(f"\n  ❌ Training data not found: {train_jsonl}")
        print("  Please run `python cli.py prepare` first.")
        sys.exit(1)

    output_dir = resolve_path(os.path.join("output", args.experiment_name))
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(output_dir, "training_config.json")
    config_data = {
        "speaker_name": args.speaker_name,
        "init_model": args.init_model,
        "model_source": args.model_source,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "grad_acc": args.grad_acc,
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)

    use_hf = args.model_source == "HuggingFace"
    resolved_model = get_model_path(args.init_model, use_hf)

    # Start TensorBoard
    print_step("Starting TensorBoard on port 6006...")
    try:
        subprocess.check_output(["pgrep", "-f", "tensorboard --logdir logs"]).decode().strip()
    except subprocess.CalledProcessError:
        subprocess.Popen(["tensorboard", "--logdir", "logs", "--port", "6006", "--bind_all"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Run training
    print_step(f"Training started on {args.gpu}...")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu.replace("cuda:", "") if args.gpu != "cpu" else ""

    from sft_12hz import run_train

    consume_generator(
        run_train(
            experiment_name=args.experiment_name,
            init_model_path=resolved_model,
            output_model_path=output_dir,
            train_jsonl=train_jsonl,
            speaker_name=args.speaker_name,
            batch_size=args.batch_size,
            lr=args.lr,
            num_epochs=args.epochs,
            gradient_accumulation_steps=args.grad_acc,
            resume_from_checkpoint="latest",
        )
    )

    print_header("✅ Training Complete!")
    print(f"  Output: {output_dir}")


def cmd_infer(args):
    """Run inference on a trained checkpoint."""
    import torch
    import soundfile as sf

    print_header("🔊 Qwen3-TTS Inference")
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Speaker      : {args.speaker}")
    print(f"  Language     : {args.language}")
    print(f"  Instruct     : {args.instruct}")
    print(f"  GPU Device   : {args.gpu}")
    print(f"  Output File  : {args.output}")
    print(f"  Text         : {args.text[:80]}{'...' if len(args.text) > 80 else ''}")

    checkpoint_path = resolve_path(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        # Try resolving via get_model_path if not a direct file path
        checkpoint_path = get_model_path(args.checkpoint, use_hf=False)
        if not os.path.exists(checkpoint_path):
            print(f"\n  ❌ Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
    
    args.checkpoint = checkpoint_path

    print_step("Loading model...")
    start = time.time()

    from qwen_tts import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=args.gpu,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if "cuda" in args.gpu else None,
    )
    print(f"    Model loaded in {time.time() - start:.2f}s")

    print_step("Synthesizing audio...")
    start = time.time()

    wavs, sr = tts.generate_custom_voice(
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        instruct=args.instruct
    )
    print(f"    Generation completed in {time.time() - start:.2f}s")

    sf.write(args.output, wavs[0], sr)
    print_header("✅ Inference Complete!")
    print(f"  Saved to: {args.output}")


def cmd_query(args):
    """Query supported speakers and languages for a model."""
    import torch
    print_header("🔍 Qwen3-TTS Model Query")
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  GPU Device   : {args.gpu}")

    if not os.path.exists(args.checkpoint):
        resolved = get_model_path(args.checkpoint, use_hf=False)
        if not os.path.exists(resolved):
            print(f"\n  ❌ Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        args.checkpoint = resolved

    print_step("Loading model...")
    from qwen_tts import Qwen3TTSModel
    tts = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=args.gpu,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if "cuda" in args.gpu else None,
    )

    print_step("Querying capabilities...")
    speakers = []
    languages = []
    if hasattr(tts, 'get_supported_speakers'):
        speakers = tts.get_supported_speakers()
    if hasattr(tts, 'get_supported_languages'):
        languages = tts.get_supported_languages()

    print(f"\n  🔊 Supported Speakers ({len(speakers)}):")
    if speakers:
        for s in speakers:
            print(f"    - {s}")
    else:
        print("    (None or standard model)")

    print(f"\n  🌐 Supported Languages ({len(languages)}):")
    if languages:
        for l in languages:
            print(f"    - {l}")
    else:
        print("    (None or default)")
    print()


# ─────────────────── Main Entry ─────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="🎙️ Qwen3-TTS Easy Finetuning — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py prepare --input_dir /workspace/raw-dataset --speaker_name my_speaker
  python cli.py train   --experiment_name exp1 --speaker_name my_speaker
  python cli.py infer   --checkpoint output/exp1/checkpoint-epoch-1 --text "Hello world"
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── prepare ──
    p_prepare = subparsers.add_parser("prepare", help="Run full data preparation pipeline (Steps 1-3)")
    p_prepare.add_argument("--input_dir", type=str, required=True, help="Directory containing raw .wav files")
    p_prepare.add_argument("--speaker_name", type=str, required=True, help="Unique speaker/dataset name")
    p_prepare.add_argument("--experiment_name", type=str, required=True, help="Experiment name for logs")
    p_prepare.add_argument("--ref_audio", type=str, default=None, help="Path to reference audio file (optional)")
    p_prepare.add_argument("--asr_model", type=str, default="Qwen/Qwen3-ASR-1.7B", help="ASR model ID")
    p_prepare.add_argument("--batch_size", type=int, default=16, help="ASR batch size")
    p_prepare.add_argument("--model_source", type=str, choices=["ModelScope", "HuggingFace"], default="ModelScope", help="Model download source")
    p_prepare.add_argument("--gpu", type=str, default="cuda:0", help="GPU device (e.g., cuda:0, cpu)")

    # ── split (Step 1) ──
    p_split = subparsers.add_parser("split", help="Step 1: Audio Split & Resample")
    p_split.add_argument("--input_dir", type=str, required=True)
    p_split.add_argument("--speaker_name", type=str, required=True)
    p_split.add_argument("--ref_audio", type=str, default=None)

    # ── asr (Step 2) ──
    p_asr = subparsers.add_parser("asr", help="Step 2: ASR Transcription & Cleaning")
    p_asr.add_argument("--speaker_name", type=str, required=True)
    p_asr.add_argument("--asr_model", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p_asr.add_argument("--batch_size", type=int, default=16)
    p_asr.add_argument("--model_source", type=str, choices=["ModelScope", "HuggingFace"], default="ModelScope")
    p_asr.add_argument("--gpu", type=str, default="cuda:0")

    # ── tokenize (Step 3) ──
    p_tokenize = subparsers.add_parser("tokenize", help="Step 3: Data Tokenization")
    p_tokenize.add_argument("--speaker_name", type=str, required=True)
    p_tokenize.add_argument("--experiment_name", type=str, required=True, help="Experiment name for saving logs/codes")
    p_tokenize.add_argument("--gpu", type=str, default="cuda:0")

    # ── train ──
    p_train = subparsers.add_parser("train", help="Run fine-tuning training")
    p_train.add_argument("--experiment_name", type=str, required=True, help="Name for this experiment")
    p_train.add_argument("--speaker_name", type=str, required=True, help="Speaker name (must match prepared data)")
    p_train.add_argument("--init_model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", help="Base model ID")
    p_train.add_argument("--model_source", type=str, choices=["ModelScope", "HuggingFace"], default="HuggingFace", help="Model download source")
    p_train.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    p_train.add_argument("--lr", type=float, default=1e-7, help="Learning rate")
    p_train.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    p_train.add_argument("--grad_acc", type=int, default=4, help="Gradient accumulation steps")
    p_train.add_argument("--gpu", type=str, default="cuda:0", help="GPU device")

    # ── infer ──
    p_infer = subparsers.add_parser("infer", help="Run inference on a trained checkpoint")
    p_infer.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint directory")
    p_infer.add_argument("--speaker", type=str, default="my_speaker", help="Speaker name used in training")
    p_infer.add_argument("--language", type=str, default="English", help="Language for synthesis")
    p_infer.add_argument("--instruct", type=str, default=None, help="Optional instruct for synthesis")
    p_infer.add_argument("--text", type=str, required=True, help="Text to synthesize")
    p_infer.add_argument("--output", type=str, default="output.wav", help="Output audio file path")
    p_infer.add_argument("--gpu", type=str, default="cuda:0", help="GPU device")

    # ── query ──
    p_query = subparsers.add_parser("query", help="Query supported speakers and languages")
    p_query.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    p_query.add_argument("--gpu", type=str, default="cuda:0", help="GPU device")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "split":
        cmd_split(args)
    elif args.command == "asr":
        cmd_asr(args)
    elif args.command == "tokenize":
        cmd_tokenize(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
