import os
import sys
import json
import glob
import torch
import gc
from tqdm import tqdm
from qwen_asr import Qwen3ASRModel

from utils import get_project_root

def log_progress(progress, desc):
    print(json.dumps({"type": "progress", "progress": progress, "desc": desc}), flush=True)

def log_done(msg):
    print(json.dumps({"type": "done", "msg": msg}), flush=True)

def log_error(msg):
    print(json.dumps({"type": "error", "msg": msg}), flush=True)

def run_step_2(input_dir, ref_audio, output_jsonl, model_id="Qwen/Qwen3-ASR-1.7B", batch_size=16):
    try:
        yield {"type": "progress", "progress": 0.01, "desc": f"Loading ASR Model: {model_id}..."}
        
        kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
        asr_model = Qwen3ASRModel.from_pretrained(model_id, **kwargs)
        
        wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
        if not wav_files:
            yield {"type": "error", "msg": f"No .wav files found in {input_dir}"}
            return
            
        final_entries = []
        total_batches = len(wav_files) // batch_size + (1 if len(wav_files) % batch_size else 0)
        
        yield {"type": "progress", "progress": 0.1, "desc": "Starting transcription..."}
        
        root_dir = get_project_root()
        for batch_idx, i in enumerate(range(0, len(wav_files), batch_size)):
            progress_pct = 0.1 + 0.85 * (batch_idx / max(total_batches, 1))
            yield {"type": "progress", "progress": progress_pct, "desc": f"Transcribing batch {batch_idx+1}/{total_batches}..."}
            
            batch_paths = wav_files[i : i + batch_size]
            try:
                results = asr_model.transcribe(audio=batch_paths)
                for path, res in zip(batch_paths, results):
                    text = res.text
                    if not text: continue
                    cleaned_text = text.strip()
                    if not cleaned_text: continue
                    
                    final_entries.append({
                        "audio": os.path.relpath(path, start=root_dir),
                        "text": cleaned_text,
                        "ref_audio": os.path.relpath(ref_audio, start=root_dir) if ref_audio else ""
                    })
            except Exception as e:
                yield {"type": "error", "msg": f"Error transcribing batch {i}: {e}"}
                continue
                
        yield {"type": "progress", "progress": 0.98, "desc": "Writing JSONL data..."}
        with open(output_jsonl, "w", encoding="utf-8") as f_out:
            for entry in final_entries:
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
        yield {"type": "done", "msg": f"Successfully transcribed {len(final_entries)} segments to {output_jsonl}."}

    except Exception as e:
        yield {"type": "error", "msg": f"Unhandled exception in step 2: {str(e)}"}
    finally:
        if 'asr_model' in locals():
            del asr_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

