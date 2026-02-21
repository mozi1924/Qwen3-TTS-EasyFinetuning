import os
import json
import torch
import glob
import re
from pydub import AudioSegment, silence
from tqdm import tqdm
from modelscope import snapshot_download
from qwen_asr import Qwen3ASRModel

def resample_audio(src_path, dest_path, target_sr=24000):
    if not os.path.exists(src_path):
        return False
    try:
        audio = AudioSegment.from_file(src_path)
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
        audio.export(dest_path, format="wav")
        return True
    except Exception as e:
        print(f"Error converting audio {src_path}: {e}")
        return False

def split_audio(audio_path, output_dir_base, filename_prefix, max_duration_ms=15000):
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return []
        
    if len(audio) <= max_duration_ms:
        out_name = f"{filename_prefix}.wav"
        out_path = os.path.join(output_dir_base, out_name)
        if not os.path.exists(out_path):
            audio.export(out_path, format="wav")
        return [out_path]
    
    chunks = silence.split_on_silence(audio, min_silence_len=500, silence_thresh=-40, keep_silence=200)
    segment_paths = []
    
    if not chunks:
        for i in range(0, len(audio), max_duration_ms):
            chunk = audio[i:i+max_duration_ms]
            if len(chunk) < 500: continue
            out_name = f"{filename_prefix}_cut{i:03d}.wav"
            out_path = os.path.join(output_dir_base, out_name)
            if not os.path.exists(out_path):
                chunk.export(out_path, format="wav")
            segment_paths.append(out_path)
    else:
        for i, chunk in enumerate(chunks):
            if len(chunk) < 500: continue
            if len(chunk) > max_duration_ms:
                for j in range(0, len(chunk), max_duration_ms):
                    sub_chunk = chunk[j:j+max_duration_ms]
                    if len(sub_chunk) < 500: continue
                    out_name = f"{filename_prefix}_seg{i:03d}_cut{j:03d}.wav"
                    out_path = os.path.join(output_dir_base, out_name)
                    if not os.path.exists(out_path):
                        sub_chunk.export(out_path, format="wav")
                    segment_paths.append(out_path)
            else:
                out_name = f"{filename_prefix}_seg{i:03d}.wav"
                out_path = os.path.join(output_dir_base, out_name)
                if not os.path.exists(out_path):
                    chunk.export(out_path, format="wav")
                segment_paths.append(out_path)
    return segment_paths

def run_pipeline(input_dir, ref_audio, output_dir, model_id="Qwen/Qwen3-ASR-1.7B", batch_size=16):
    audio_out_dir = os.path.join(output_dir, "audio")
    audio_24k_dir = os.path.join(output_dir, "audio_24k")
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(audio_24k_dir, exist_ok=True)
    
    final_jsonl = os.path.join(output_dir, "tts_train.jsonl")
    
    print("Preparing Reference Audio...")
    ref_24k_path = os.path.join(audio_24k_dir, "ref_24k.wav")
    if not os.path.exists(ref_audio) and ref_audio != "":
        return False, f"Reference audio not found: {ref_audio}"
    if ref_audio:
        resample_audio(ref_audio, ref_24k_path)
    
    print(f"Loading ASR Model: {model_id}")
    model_dir = snapshot_download(model_id)
    kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    asr_model = Qwen3ASRModel.from_pretrained(model_dir, **kwargs)
    
    wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    all_segments = []
    
    print("Splitting audio files...")
    for wav_path in tqdm(wav_files, desc="Splitting"):
        prefix = os.path.splitext(os.path.basename(wav_path))[0]
        segs = split_audio(wav_path, audio_out_dir, prefix)
        all_segments.extend([os.path.abspath(s) for s in segs])
    
    final_entries = []
    print("Transcribing and processing data...")
    for i in tqdm(range(0, len(all_segments), batch_size), desc="ASR processing"):
        batch_paths = all_segments[i : i + batch_size]
        try:
            results = asr_model.transcribe(audio=batch_paths)
            for path, res in zip(batch_paths, results):
                text = res.text
                if not text: continue
                # cleaning text directly
                cleaned_text = text.strip()
                if not cleaned_text: continue
                
                # Resample this segment
                filename = os.path.basename(path)
                dest_audio = os.path.join(audio_24k_dir, filename)
                if not os.path.exists(dest_audio):
                    resample_audio(path, dest_audio)
                
                final_entries.append({
                    "audio": os.path.abspath(dest_audio),
                    "text": cleaned_text,
                    "ref_audio": os.path.abspath(ref_24k_path) if ref_audio else ""
                })
        except Exception as e:
            print(f"Error transcribing batch {i}: {e}")
            continue

    with open(final_jsonl, "w", encoding="utf-8") as f_out:
        for entry in final_entries:
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    return True, f"Successfully processed {len(final_entries)} segments. Saved to {final_jsonl}"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--ref_audio", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    success, msg = run_pipeline(args.input_dir, args.ref_audio, args.output_dir, args.model_id, args.batch_size)
    print(msg)
