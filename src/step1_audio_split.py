import os
import sys
import json
import glob
from pydub import AudioSegment, silence
from tqdm import tqdm

def log_progress(progress, desc):
    print(json.dumps({"type": "progress", "progress": progress, "desc": desc}), flush=True)

def log_done(msg):
    print(json.dumps({"type": "done", "msg": msg}), flush=True)

def log_error(msg):
    print(json.dumps({"type": "error", "msg": msg}), flush=True)

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
        log_error(f"Error converting audio {src_path}: {e}")
        return False

def split_audio(audio_path, output_dir_base, filename_prefix, max_duration_ms=15000, min_duration_ms=1000):
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        log_error(f"Error reading {audio_path}: {e}")
        return []

    thresh = audio.dBFS - 16 if audio.dBFS > -50 else -40
    
    chunks = silence.split_on_silence(
        audio, 
        min_silence_len=400, 
        silence_thresh=thresh,
        keep_silence=200
    )
    
    if not chunks:
        chunks = [audio]
        
    segment_paths = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk) < min_duration_ms:
            continue
            
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

def run_step_1(input_dir, output_dir, ref_audio=None):
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle reference audio if provided
        if ref_audio and os.path.exists(ref_audio):
            yield {"type": "progress", "progress": 0.0, "desc": "Processing reference audio..."}
            ref_ext = os.path.splitext(ref_audio)[1]
            ref_24k_path = os.path.join(output_dir, "ref_24k.wav")
            resample_audio(ref_audio, ref_24k_path)
            yield {"type": "progress", "progress": 0.05, "desc": f"Reference audio saved to {ref_24k_path}"}

        wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
        if not wav_files:
            yield {"type": "error", "msg": f"No .wav files found in {input_dir}"}
            return
            
        all_segments = []
        yield {"type": "progress", "progress": 0.1, "desc": "Splitting audio files..."}
        
        for idx, wav_path in enumerate(wav_files):
            yield {"type": "progress", "progress": 0.1 + 0.5 * (idx / len(wav_files)), "desc": f"Splitting {idx+1}/{len(wav_files)}"}
            prefix = os.path.splitext(os.path.basename(wav_path))[0]
            segs = split_audio(wav_path, output_dir, prefix)
            all_segments.extend([os.path.abspath(s) for s in segs])
            
        # Resample all segments to 24kHz in place
        for idx, seg in enumerate(all_segments):
            yield {"type": "progress", "progress": 0.6 + 0.3 * (idx / len(all_segments)), "desc": f"Resampling segment {idx+1}/{len(all_segments)}"}
            resample_audio(seg, seg) # Overwrite with 24kHz
            
        yield {"type": "done", "msg": f"Successfully split and resampled {len(all_segments)} segments."}
        
    except Exception as e:
        yield {"type": "error", "msg": f"Unhandled exception in step 1: {str(e)}"}


