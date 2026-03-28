import os
import sys
import json
import glob
from pydub import AudioSegment, silence
from tqdm import tqdm
import concurrent.futures
import multiprocessing

EDGE_SILENCE_MS = 200
FADE_MS = 40
SILENCE_SCAN_STEP_MS = 10

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

def detect_leading_silence_ms(audio, silence_thresh, chunk_size=SILENCE_SCAN_STEP_MS):
    trim_ms = 0
    while trim_ms < len(audio):
        if audio[trim_ms:trim_ms + chunk_size].dBFS > silence_thresh:
            break
        trim_ms += chunk_size
    return min(trim_ms, len(audio))

def strip_edge_silence(audio, silence_thresh):
    if len(audio) == 0:
        return audio

    leading_ms = detect_leading_silence_ms(audio, silence_thresh)
    trailing_ms = detect_leading_silence_ms(audio.reverse(), silence_thresh)
    end_ms = len(audio) - trailing_ms
    if end_ms <= leading_ms:
        return audio
    return audio[leading_ms:end_ms]

def center_and_smooth_segment(audio, silence_thresh, pad_silence_ms=EDGE_SILENCE_MS, fade_in_ms=0, fade_out_ms=0):
    trimmed = strip_edge_silence(audio, silence_thresh)
    if len(trimmed) == 0:
        trimmed = audio

    if fade_in_ms > 0:
        trimmed = trimmed.fade_in(min(fade_in_ms, len(trimmed)))
    if fade_out_ms > 0:
        trimmed = trimmed.fade_out(min(fade_out_ms, len(trimmed)))

    edge_silence = AudioSegment.silent(duration=pad_silence_ms, frame_rate=trimmed.frame_rate)
    return edge_silence + trimmed + edge_silence

def smooth_hard_cut_segment(audio, fade_in_ms=0, fade_out_ms=0):
    smoothed = audio
    if fade_in_ms > 0:
        smoothed = smoothed.fade_in(min(fade_in_ms, len(smoothed)))
    if fade_out_ms > 0:
        smoothed = smoothed.fade_out(min(fade_out_ms, len(smoothed)))
    return smoothed

def split_audio(audio_path, output_dir_base, filename_prefix, max_duration_ms=15000, min_duration_ms=1000, target_sr=24000):
    try:
        audio = AudioSegment.from_file(audio_path)
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
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
            cut_points = list(range(0, len(chunk), max_duration_ms))
            for cut_idx, j in enumerate(cut_points):
                sub_chunk = chunk[j:j+max_duration_ms]
                if len(sub_chunk) < min_duration_ms:
                    continue

                sub_chunk = smooth_hard_cut_segment(
                    sub_chunk,
                    fade_in_ms=FADE_MS if cut_idx > 0 else 0,
                    fade_out_ms=FADE_MS if cut_idx < len(cut_points) - 1 else 0,
                )
                
                out_name = f"{filename_prefix}_seg{i:03d}_cut{j:03d}.wav"
                out_path = os.path.join(output_dir_base, out_name)
                if not os.path.exists(out_path):
                    sub_chunk.export(out_path, format="wav")
                segment_paths.append(out_path)
        else:
            chunk = center_and_smooth_segment(chunk, silence_thresh=thresh)
            out_name = f"{filename_prefix}_seg{i:03d}.wav"
            out_path = os.path.join(output_dir_base, out_name)
            if not os.path.exists(out_path):
                chunk.export(out_path, format="wav")
            segment_paths.append(out_path)
            
    return segment_paths

def run_step_1(input_dir, output_dir, ref_audio=None, num_threads=6, skip_split=False):
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
        wav_files = [f for f in wav_files if os.path.basename(f) not in ["ref.wav", "ref_24k.wav"]]
        if not wav_files:
            yield {"type": "error", "msg": f"No .wav files found in {input_dir}"}
            return
            
        all_segments = []
        action_desc = "Resampling" if skip_split else "Splitting and resampling"
        yield {"type": "progress", "progress": 0.1, "desc": f"{action_desc} {len(wav_files)} files using {num_threads} threads..."}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            if skip_split:
                futures = {
                    executor.submit(
                        resample_audio,
                        wav_path,
                        os.path.join(output_dir, os.path.basename(wav_path)),
                    ): wav_path
                    for wav_path in wav_files
                }
            else:
                futures = {
                    executor.submit(split_audio, wav_path, output_dir, os.path.splitext(os.path.basename(wav_path))[0]): wav_path
                    for wav_path in wav_files
                }
            
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                yield {"type": "progress", "progress": 0.1 + 0.8 * (idx / max(1, len(wav_files))), "desc": f"Processing {idx+1}/{len(wav_files)}"}
                try:
                    result = future.result()
                    src_path = futures[future]
                    if skip_split:
                        if result:
                            all_segments.append(os.path.abspath(os.path.join(output_dir, os.path.basename(src_path))))
                    else:
                        all_segments.extend([os.path.abspath(s) for s in result])
                except Exception as e:
                    yield {"type": "error", "msg": f"Error in processing: {str(e)}"}
            
        yield {"type": "done", "msg": f"Successfully split and resampled {len(all_segments)} segments."}
        
    except Exception as e:
        yield {"type": "error", "msg": f"Unhandled exception in step 1: {str(e)}"}
