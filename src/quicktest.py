import argparse
import time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

@torch.no_grad()
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/checkpoint-epoch-0", help="Path to fine-tuned model")
    parser.add_argument("--text", type=str, default="Hello, this is a test from my custom voice.", help="Text to synthesize")
    parser.add_argument("--speaker", type=str, default="my_speaker", help="Speaker name used in training")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} on {args.device}...")
    start_time = time.time()
    
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

    print(f"Synthesizing text: '{args.text}' with speaker '{args.speaker}'...")
    start_time = time.time()
    
    wavs, sr = tts.generate_custom_voice(
        text=args.text,
        speaker=args.speaker,
    )
    print(f"Generation completed in {time.time() - start_time:.2f} seconds.")

    sf.write(args.output, wavs[0], sr)
    print(f"Done! Result saved to {args.output}")

