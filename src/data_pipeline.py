import os
import json
import torch
import glob
import re
from pydub import AudioSegment, silence
from tqdm import tqdm
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

def split_audio(audio_path, output_dir_base, filename_prefix, max_duration_ms=15000, min_duration_ms=1000):
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return []

    # Strip leading/trailing silence & split on mid-silences for ALL chunks
    # We use a dynamic threshold: audio.dBFS - 16 dB often works great
    thresh = audio.dBFS - 16 if audio.dBFS > -50 else -40
    
    chunks = silence.split_on_silence(
        audio, 
        min_silence_len=400, 
        silence_thresh=thresh,
        keep_silence=200
    )
    
    if not chunks:
        # Fallback to the whole audio if no silence could be detected
        chunks = [audio]
        
    segment_paths = []
    
    for i, chunk in enumerate(chunks):
        # Drop segments that are too short or empty
        if len(chunk) < min_duration_ms:
            continue
            
        # Hard cut segments that are still too long
        if len(chunk) > max_duration_ms:
            for j in range(0, len(chunk), max_duration_ms):
                sub_chunk = chunk[j:j+max_duration_ms]
                if len(sub_chunk) < min_duration_ms:
                    continue
                
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

def run_pipeline(input_dir, ref_audio, output_dir, model_id="Qwen/Qwen3-ASR-1.7B", batch_size=16, progress=None):
    audio_out_dir = os.path.join(output_dir, "audio")
    audio_24k_dir = os.path.join(output_dir, "audio_24k")
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(audio_24k_dir, exist_ok=True)
    
    final_jsonl = os.path.join(output_dir, "tts_train.jsonl")
    
    if progress: progress(0.01, desc="Preparing Reference Audio...")
    print("Preparing Reference Audio...")
    ref_24k_path = os.path.join(audio_24k_dir, "ref_24k.wav")
    if not os.path.exists(ref_audio) and ref_audio != "":
        return False, f"Reference audio not found: {ref_audio}"
    if ref_audio:
        resample_audio(ref_audio, ref_24k_path)
    
    if progress: progress(0.1, desc=f"Loading Model: {model_id} (Downloading might take a while...)")
    print(f"Loading ASR Model: {model_id}")
    kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    asr_model = Qwen3ASRModel.from_pretrained(model_id, **kwargs)
    
    wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    all_segments = []
    
    if progress: progress(0.3, desc="Splitting audio files...")
    print("Splitting audio files...")
    for idx, wav_path in enumerate(tqdm(wav_files, desc="Splitting")):
        if progress: progress(0.3 + 0.2 * (idx / len(wav_files)), desc=f"Splitting {idx+1}/{len(wav_files)}")
        prefix = os.path.splitext(os.path.basename(wav_path))[0]
        segs = split_audio(wav_path, audio_out_dir, prefix)
        all_segments.extend([os.path.abspath(s) for s in segs])
    
    final_entries = []
    if progress: progress(0.5, desc="Transcribing and processing data...")
    print("Transcribing and processing data...")
    total_batches = len(all_segments) // batch_size + (1 if len(all_segments) % batch_size else 0)
    for batch_idx, i in enumerate(tqdm(range(0, len(all_segments), batch_size), desc="ASR processing")):
        if progress: progress(0.5 + 0.49 * (batch_idx / max(total_batches, 1)), desc=f"ASR transcribing batch {batch_idx+1}/{total_batches}")
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
            
    if progress: progress(1.0, desc="Pipeline Completed!")
    return True, f"Successfully processed {len(final_entries)} segments. Saved to {final_jsonl}"

