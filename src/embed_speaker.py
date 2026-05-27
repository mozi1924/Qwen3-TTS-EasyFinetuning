#!/usr/bin/env python3
"""
Pre-compute speaker embeddings from reference audio(s).

This script extracts speaker embeddings using the Base model's speaker_encoder
and saves them as safetensors. The training loop can then load these pre-computed
embeddings directly, avoiding redundant GPU computation in every training step.

Usage:
  # Per-speaker ref from JSONL (default)
  python src/embed_speaker.py --base_model Qwen/Qwen3-TTS-12Hz-0.6B-Base

  # Average all audio per speaker
  python src/embed_speaker.py --base_model Qwen/Qwen3-TTS-12Hz-1.7B-Base --mode avg_all

  # Use HuggingFace instead of the project's default model source
  python src/embed_speaker.py --base_model Qwen/Qwen3-TTS-12Hz-0.6B-Base --model_source HuggingFace

  # Custom ref audio for a single speaker
  python src/embed_speaker.py --speaker my_speaker --ref /path/to/ref.wav

  # Multi-ref averaging
  python src/embed_speaker.py --speaker my_speaker --ref a.wav,b.wav,c.wav

Output: final-dataset/{speaker}/speaker_emb.safetensors (1024-dim tensor)
"""

import argparse
import gc
import json
import os
import sys

import torch
import librosa
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from utils import get_model_path


def get_runtime_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_speaker_encoder(model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base", device="cuda:0"):
    """Load Base model just for its speaker_encoder."""
    base = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="flash_attention_2" if device.startswith("cuda") else "eager",
    )
    se = base.model.speaker_encoder.to(device)
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return se


def extract_embedding(se, audio_path, device):
    """Extract 1024-dim speaker embedding from audio file."""
    audio, sr = librosa.load(audio_path, sr=24000, mono=True)
    mel = mel_spectrogram(
        torch.from_numpy(audio).unsqueeze(0).to(device),
        n_fft=1024, num_mels=128, sampling_rate=24000,
        hop_size=256, win_size=1024, fmin=0, fmax=12000,
    ).transpose(1, 2)
    mel = mel.to(torch.bfloat16 if device.startswith("cuda") else torch.float32)
    with torch.no_grad():
        emb = se(mel).detach()  # [1, 1024]
    return emb[0].cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                        help="Base model HF ID for speaker_encoder extraction")
    parser.add_argument("--model_source", default="ModelScope", choices=["HuggingFace", "ModelScope"],
                        help="Model download source, consistent with the rest of the project")
    parser.add_argument("--mode", default="ref", choices=["ref", "avg_all"],
                        help="ref=use ref_audio from JSONL, avg_all=average all samples per speaker")
    parser.add_argument("--speaker", default=None, help="Process single speaker")
    parser.add_argument("--ref", default=None, help="Custom ref audio(s), comma-separated")
    parser.add_argument("--output", default=None, help="Custom output path")
    args = parser.parse_args()

    device = get_runtime_device()
    use_hf = args.model_source == "HuggingFace"
    resolved_base_model = get_model_path(args.base_model, use_hf=use_hf)
    print(f"Loading speaker_encoder from {args.base_model}")
    print(f"Resolved model path: {resolved_base_model}")
    se = load_speaker_encoder(resolved_base_model, device=device)

    dataset_path = "final-dataset"
    if not os.path.isdir(dataset_path):
        print("final-dataset/ not found. Run the data pipeline first.")
        sys.exit(1)

    speakers = [args.speaker] if args.speaker else [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]

    for spk in speakers:
        # Embeddings go alongside speaker data in final-dataset/
        spk_dir = os.path.join(dataset_path, spk)
        jsonl = os.path.join(spk_dir, "tts_train.jsonl")
        audio_dir = os.path.join(spk_dir, "audio_24k")
        out_path = args.output or os.path.join(spk_dir, "speaker_emb.safetensors")

        embeddings = []

        if args.ref:
            for ref_file in args.ref.split(","):
                ref_file = ref_file.strip()
                if not os.path.isabs(ref_file):
                    ref_file = os.path.join(audio_dir, ref_file)
                if os.path.exists(ref_file):
                    print(f"  {spk}: {os.path.basename(ref_file)}")
                    embeddings.append(extract_embedding(se, ref_file, device))
        elif args.mode == "avg_all":
            if os.path.exists(jsonl):
                with open(jsonl) as f:
                    entries = [json.loads(line) for line in f]
                for entry in entries:
                    audio_path = entry.get("audio")
                    if audio_path and os.path.exists(audio_path):
                        embeddings.append(extract_embedding(se, audio_path, device))
                print(f"  {spk}: averaged {len(entries)} samples")
            else:
                print(f"  {spk}: no JSONL, skipping")
                continue
        else:
            # Default: use ref_audio from JSONL (set by Step 2 pipeline)
            if os.path.exists(jsonl):
                with open(jsonl) as f:
                    entries = [json.loads(line) for line in f]
                if entries:
                    ref = entries[0].get("ref_audio")
                    if ref and os.path.exists(ref):
                        print(f"  {spk}: {os.path.basename(ref)}")
                        embeddings.append(extract_embedding(se, ref, device))
                    else:
                        print(f"  {spk}: ref_audio not found in JSONL, skipping")
                        continue
                else:
                    print(f"  {spk}: JSONL is empty, skipping")
                    continue
            else:
                print(f"  {spk}: no JSONL, skipping")
                continue

        if embeddings:
            avg_emb = torch.stack(embeddings).mean(dim=0).squeeze(0)  # [D]
            from safetensors.torch import save_file
            save_file({"emb": avg_emb}, out_path)
            print(f"  → saved {out_path} (norm={avg_emb.norm():.2f})")
        else:
            print(f"  {spk}: no embeddings extracted")

    del se
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
