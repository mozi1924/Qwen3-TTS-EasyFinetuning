import os
import glob
import subprocess
import gradio as gr
import json
import threading
from step1_audio_split import run_step_1 as internal_run_step_1
from step2_asr_clean import run_step_2 as internal_run_step_2
from prepare_data import run_prepare as internal_run_prepare
from sft_12hz import run_train as internal_run_train

import time
import torch
import gc
import sys
from utils import get_model_path, get_model_local_dir, get_project_root, resolve_path, resolve_speaker_choice
from webui_training import (
    get_checkpoints,
    normalize_speaker_name,
    normalize_resume_checkpoint,
    save_training_config,
    build_training_kwargs,
    append_log,
    handle_training_message,
    load_experiment_config,
    on_new_experiment,
    stream_worker_updates,
    get_deeplink_state,
    run_with_polling,
)

def get_build_info():
    try:
        with open("build_info.json", "r") as f:
            return json.load(f)
    except:
        return None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tqdm

# Configure HF cache directory from environment variables ONLY ONCE
hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface")
print(f"HF cache initialized to: {hf_home}")

# ----------------- Tqdm Wrapper for Gradio -----------------
class GradioTqdm(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        self._gr_progress = kwargs.pop("gr_progress", None)
        super().__init__(*args, **kwargs)
        
    def update(self, n=1):
        displayed = super().update(n)
        if self._gr_progress and self.total:
            # We use a slight delta to avoid flickering
            self._gr_progress(self.n / self.total, desc=f"Downloading... {self.desc or ''}")
        return displayed

# Global references
global_tts_model = None
global_tts_model_path = None
global_tts_device = None
global_training_process = None

# Removed redundant models_config.json loading as it is not present and presets are hardcoded below


def get_gpus():
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpus.append(f"cuda:{i}")
    if not gpus:
        gpus = ["cpu"]
    return gpus

def get_datasets():
    dataset_path = resolve_path("final-dataset")
    if not os.path.exists(dataset_path): return []
    return [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

def get_raw_datasets():
    base = "raw-dataset"
    if not os.path.exists(base): 
        os.makedirs(base, exist_ok=True)
    dirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    return sorted([base] + dirs)

def get_ref_audios():
    base = "raw-dataset"
    if not os.path.exists(base): return ["raw-dataset/ref.wav"]
    
    paths = ["raw-dataset/ref.wav"] # Always show default as an option
    
    # Check first-level subdirectories for ref.wav
    for d in os.listdir(base):
        dir_path = os.path.join(base, d)
        if os.path.isdir(dir_path):
            ref_sub = os.path.join(dir_path, "ref.wav")
            # We don't necessarily need to check if exists, 
            # as the user might want to select/type it then put the file.
            # But the user said "命名为ref.wav放进去", so showing existing ones is better.
            if os.path.exists(ref_sub):
                paths.append(ref_sub)
                
    return sorted(list(set(paths)))

# Removed local get_model_path as it is now imported from utils

import multiprocessing as mp
import queue
import inspect

global_training_stop_event = None

def _run_worker(func, q, stop_event, env_vars, args, kwargs):
    try:
        import os
        if env_vars:
            os.environ.update(env_vars)
            
        if stop_event is not None:
            kwargs['stop_event'] = stop_event
        for item in func(*args, **kwargs):
            q.put(item)
    except Exception as e:
        import traceback
        q.put({"type": "error", "msg": f"Worker Exception: {str(e)}\n{traceback.format_exc()}"})
    finally:
        q.put(None)

def stream_isolated(func, *args, **kwargs):
    global global_training_stop_event
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    
    stop_event = None
    if getattr(func, '__name__', '') == 'run_train':
        stop_event = ctx.Event()
        global_training_stop_event = stop_event
        
    import os
    env_vars = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
        
    p = ctx.Process(target=_run_worker, args=(func, q, stop_event, env_vars, args, kwargs))
    p.start()
    
    while True:
        try:
            item = q.get(timeout=0.2)
            if item is None:
                break
            yield item
        except queue.Empty:
            if not p.is_alive():
                break
                
    p.join()

# ----------------- Step 1: Audio Split -----------------
def _normalize_single_speaker_name(speaker_name, error_message):
    speaker = speaker_name.strip() if isinstance(speaker_name, str) else ""
    if not speaker:
        return None, error_message
    return speaker, None


def run_step_1(input_dir, speaker_name, ref_audio, num_threads=6, skip_split=False, progress=gr.Progress()):
    speaker, error = _normalize_single_speaker_name(speaker_name, "Please specify a Speaker Name first.")
    if error:
        yield error
        return

    output_dir = resolve_path(os.path.join("final-dataset", speaker, "audio_24k"))
    ref_path = resolve_path(ref_audio) if ref_audio else None
    stream = stream_isolated(
        internal_run_step_1,
        resolve_path(input_dir),
        output_dir,
        ref_path,
        num_threads=num_threads,
        skip_split=skip_split,
    )
    yield from stream_worker_updates(stream, progress)

# ----------------- Step 2: ASR Transcription -----------------
def run_step_2(speaker_name, asr_model, asr_source, gpu_id, progress=gr.Progress()):
    speaker, error = _normalize_single_speaker_name(speaker_name, "Please specify a Speaker Name first.")
    if error:
        yield error
        return

    speaker_dir = resolve_path(os.path.join("final-dataset", speaker))
    input_dir = os.path.join(speaker_dir, "audio_24k")
    output_jsonl = os.path.join(speaker_dir, "tts_train.jsonl")
    ref_24k = os.path.join(input_dir, "ref_24k.wav")
    ref_path = ref_24k if os.path.exists(ref_24k) else ""

    if not os.path.exists(input_dir):
        yield f"Error: Directory {input_dir} not found. Please run Step 1 first."
        return

    use_hf = asr_source == "HuggingFace"
    resolved_model_id = get_model_path(asr_model, use_hf=use_hf)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id.replace("cuda:", "") if gpu_id != "cpu" else ""
    stream = stream_isolated(internal_run_step_2, input_dir, ref_path, output_jsonl, resolved_model_id, batch_size=16)
    yield from stream_worker_updates(stream, progress)

# ----------------- Step 3: Tokenization -----------------
def run_step_3(speaker_name, experiment_name, gpu_id, progress=gr.Progress()):
    if isinstance(speaker_name, list):
        speaker_names = [s.strip() for s in speaker_name if s.strip()]
    elif isinstance(speaker_name, str):
        speaker_names = [s.strip() for s in speaker_name.split(',') if s.strip()]
    else:
        speaker_names = []

    experiment_name = experiment_name.strip() if experiment_name else ""
    if not speaker_names or not experiment_name:
        yield "Please specify Speaker Name and Experiment Name."
        return

    log_dir = resolve_path(os.path.join("logs", experiment_name))
    os.makedirs(log_dir, exist_ok=True)
    output_jsonl = os.path.join(log_dir, "tts_train_with_codes.jsonl")

    if len(speaker_names) > 1:
        merged_jsonl = os.path.join(log_dir, "tts_train_merged.jsonl")
        total_merged = 0
        with open(merged_jsonl, 'w', encoding='utf-8') as f_out:
            for spk_name in speaker_names:
                speaker_dir = resolve_path(os.path.join("final-dataset", spk_name))
                input_jsonl = os.path.join(speaker_dir, "tts_train.jsonl")
                if not os.path.exists(input_jsonl):
                    yield f"Error: File {input_jsonl} not found for speaker '{spk_name}'. Please run Data Prep Step 1 & 2 first."
                    return
                with open(input_jsonl, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        entry = json.loads(line.strip())
                        entry['speaker_id'] = spk_name
                        f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        total_merged += 1
        yield f"Merged {total_merged} entries from {len(speaker_names)} speakers. Starting tokenization..."
        input_jsonl = merged_jsonl
    else:
        speaker_dir = resolve_path(os.path.join("final-dataset", speaker_names[0]))
        input_jsonl = os.path.join(speaker_dir, "tts_train.jsonl")
        if not os.path.exists(input_jsonl):
            yield f"Error: File {input_jsonl} not found. Please run Data Prep Step 1 & 2 first."
            return

    resolved_tokenizer = get_model_path("Qwen/Qwen3-TTS-Tokenizer-12Hz", use_hf=False)
    device = "cuda:0" if gpu_id != "cpu" else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id.replace("cuda:", "") if gpu_id != "cpu" else ""
    stream = stream_isolated(internal_run_prepare, device, resolved_tokenizer, input_jsonl, output_jsonl)
    yield from stream_worker_updates(stream, progress)

# ----------------- Download Model -----------------
def check_or_download_model(init_model, model_source, progress=gr.Progress()):
    use_hf = model_source == "HuggingFace"
    tokenizer_model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    try:
        progress(0.02, desc="Preparing downloads...")
        def download_base_model():
            return get_model_path(init_model, use_hf=use_hf)
        download_base_model.target_dir = get_model_local_dir(init_model)
        resolved_init_model = run_with_polling(download_base_model, progress, progress_start=0.05, progress_end=0.55, desc_prefix=f"Downloading base model {init_model}")
        def download_tokenizer_model():
            return get_model_path(tokenizer_model_id, use_hf=False)
        download_tokenizer_model.target_dir = get_model_local_dir(tokenizer_model_id)
        resolved_tokenizer = run_with_polling(download_tokenizer_model, progress, progress_start=0.60, progress_end=0.98, desc_prefix=f"Downloading tokenizer {tokenizer_model_id}")
        progress(1.0, desc="All required models are ready")
        return f"Base model ready at: {resolved_init_model}\nTokenizer ready at: {resolved_tokenizer}"
    except Exception as e:
        progress(0, desc="Download failed")
        return f"Error downloading model(s): {e}"

def check_tb():
    def is_process_running(process_name):
        try:
            output = subprocess.check_output(["pgrep", "-f", process_name]).decode().strip()
            return bool(output)
        except subprocess.CalledProcessError:
            return False
    if not is_process_running("tensorboard --logdir logs"):
        subprocess.Popen(["tensorboard", "--logdir", "logs", "--port", "6006", "--bind_all"])

def stop_tensorboard():
    try:
        subprocess.run(["pkill", "-f", "tensorboard --logdir logs"], check=False)
        return "Tensorboard server stopped."
    except Exception as e:
        return f"Error stopping Tensorboard: {e}"
# ---------------- Training -----------------
def start_training(
    experiment_name,
    speaker_name,
    init_model,
    model_source,
    batch_size,
    lr,
    epochs,
    grad_acc,
    gpu_id,
    use_experimental_speedup,
    resume_from_checkpoint,
    save_strategy,
    save_steps,
    keep_last_n_checkpoints,
    use_accelerator,
    progress=gr.Progress(),
):
    global_training_stop_event
    unload_model()
    speaker_name_str = normalize_speaker_name(speaker_name)
    if not experiment_name or not speaker_name_str:
        yield "Error: Please select or type an Experiment Name / Speaker Name.", ""
        return
    train_jsonl = resolve_path(os.path.join("logs", experiment_name, "tts_train_with_codes.jsonl"))
    if not os.path.exists(train_jsonl):
        yield f"Error: JSONL file {train_jsonl} not found. Please run tokenization (Step 1 -> 2 -> 3) first.", ""
        return
    output_dir = resolve_path(os.path.join("output", experiment_name))
    os.makedirs(output_dir, exist_ok=True)
    config_data = {
        "speaker_name": speaker_name_str,
        "init_model": init_model,
        "model_source": model_source,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "grad_acc": grad_acc,
        "use_experimental_speedup": use_experimental_speedup,
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
        "save_strategy": save_strategy,
        "save_steps": save_steps,
        "keep_last_n_checkpoints": keep_last_n_checkpoints,
        "use_accelerator": use_accelerator,
    }
    save_training_config(output_dir, config_data)
    use_hf = model_source == "HuggingFace"
    resolved_init_model = get_model_path(init_model, use_hf=use_hf)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id.replace("cuda:", "") if gpu_id != "cpu" else ""
    check_tb()
    print(f"Starting in-process training on {gpu_id}...")
    log_history = []
    last_status = "Starting..."
    final_resume = normalize_resume_checkpoint(resume_from_checkpoint)
    training_kwargs = build_training_kwargs(
        experiment_name=experiment_name,
        speaker_name_str=speaker_name_str,
        resolved_init_model=resolved_init_model,
        output_dir=output_dir,
        train_jsonl=train_jsonl,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        grad_acc=grad_acc,
        final_resume=final_resume,
        use_experimental_speedup=use_experimental_speedup,
        save_strategy=save_strategy,
        save_steps=save_steps,
        keep_last_n_checkpoints=keep_last_n_checkpoints,
        use_accelerator=use_accelerator,
    )
    gen = stream_isolated(internal_run_train, **training_kwargs)
    try:
        total_epochs = epochs or 1
        for item in gen:
            if isinstance(item, dict):
                status_update, logs_update, should_stop = handle_training_message(
                    item=item,
                    progress=progress,
                    total_epochs=total_epochs,
                    last_status=last_status,
                    log_history=log_history,
                )
                if isinstance(status_update, str):
                    last_status = status_update
                yield status_update, logs_update
                if should_stop:
                    return
            elif isinstance(item, str):
                last_status = item
                yield last_status, append_log(log_history, item)
    except Exception as e:
        yield f"Error: Unhandled exception {str(e)}", "\n".join(log_history[-30:])

def stop_training():
    global_training_stop_event
    if global_training_stop_event is not None:
        global_training_stop_event.set()
        return "Notified training process to abort safely!"
    return "No training process currently running."

def _get_model_capabilities(tts_model):
    speakers = []
    languages = []
    if hasattr(tts_model, "get_supported_speakers"):
        speakers = tts_model.get_supported_speakers()
    if hasattr(tts_model, "get_supported_languages"):
        languages = tts_model.get_supported_languages()
    return speakers, languages



def load_model(model_path, gpu_id):
    global global_tts_model, global_tts_model_path, global_tts_device, global_inference_busy
    with global_model_lock:
        if not model_path:
            unload_msg = unload_model(force=True)
            return unload_msg, [], []
        if global_inference_busy and (global_tts_model_path != model_path or global_tts_device != gpu_id):
            return "Inference is busy. Please wait before switching models.", [], []
        if global_tts_model_path == model_path and global_tts_device == gpu_id and global_tts_model is not None:
            speakers, languages = _get_model_capabilities(global_tts_model)
            return "Model already loaded.", speakers, languages
        if global_tts_model is not None:
            unload_model(force=True)
        try:
            resolved_model_path = get_model_path(model_path, use_hf=False)
            print(f"Loading {resolved_model_path} on {gpu_id}...")
            from qwen_tts import Qwen3TTSModel
            global_tts_model = Qwen3TTSModel.from_pretrained(
                resolved_model_path,
                device_map=gpu_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if "cuda" in gpu_id else None,
            )
            global_tts_model_path = model_path
            global_tts_device = gpu_id
            speakers, languages = _get_model_capabilities(global_tts_model)
            return f"Loaded {model_path} successfully.", speakers, languages
        except Exception as e:
            global_tts_model = None
            global_tts_model_path = None
            global_tts_device = None
            return f"Failed to load model: {e}", [], []


def unload_model(force=False):
    global global_tts_model, global_tts_model_path, global_tts_device, global_inference_busy
    with global_model_lock:
        if global_inference_busy and not force:
            return "Inference is running. Please wait before unloading model."
        if global_tts_model is not None:
            del global_tts_model
            global_tts_model = None
            global_tts_model_path = None
            global_tts_device = None
            gc.collect()
            torch.cuda.empty_cache()
            return "Model unloaded and VRAM cleared."
        return "No model was loaded."


def run_inference(model_path, speaker, language, text, instruct, gpu_id, progress=gr.Progress()):
    global global_tts_model, global_tts_model_path, global_inference_busy
    if not model_path:
        return None, "Please select a model checkpoint."
    if global_tts_model_path != model_path or global_tts_model is None:
        progress(0.1, desc="Loading Model into VRAM...")
        load_msg, _, _ = load_model(model_path, gpu_id)
        if "Failed" in load_msg or "busy" in load_msg.lower():
            return None, load_msg
    try:
        global_inference_busy = True
        progress(0.5, desc="Synthesizing audio...")
        import soundfile as sf
        supported_speakers, _ = _get_model_capabilities(global_tts_model)
        resolved_speaker = resolve_speaker_choice(speaker, supported_speakers)
        wavs, sr = global_tts_model.generate_custom_voice(
            text=text,
            speaker=resolved_speaker,
            language=language,
            instruct=instruct if instruct else None,
        )
        out_path = "webui_output.wav"
        sf.write(out_path, wavs[0], sr)
        progress(1.0, desc="Done!")
        return out_path, f"Inference successful. Speaker: {resolved_speaker}"
    except Exception as e:
        return None, f"Inference error: {e}"
    finally:
        global_inference_busy = False

def on_checkpoint_change(ckpt_path):
    return gr.update()

def refresh_checkpoints():
    return gr.update(choices=get_checkpoints(include_specials=False))

def refresh_datasets():
    return gr.update(choices=get_datasets())
presets = {
    "0.6B Model": { "init_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "lr": 1e-7, "epochs": 2, "batch_size": 2, "grad_acc": 4 },
    "1.7B Model": { "init_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "lr": 2e-6, "epochs": 3, "batch_size": 2, "grad_acc": 1 },
    "Latest Config": {}
}

def apply_preset(preset_name, experiment_name):
    if preset_name == "Latest Config" and experiment_name:
        config_path = os.path.join("output", experiment_name, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = json.load(f)
            return data.get("init_model"), data.get("lr"), data.get("epochs"), data.get("batch_size"), data.get("grad_acc")
            
    p = presets.get(preset_name, presets["0.6B Model"])
    return p["init_model"], p["lr"], p["epochs"], p["batch_size"], p["grad_acc"]

def get_experiments():
    if not os.path.exists("output"): return []
    return [d for d in os.listdir("output") if os.path.isdir(os.path.join("output", d))]

# ----------------- UI -----------------
css = """
/* Global Styles */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Remove gray boxes behind text and group containers */
.gr-group, .gr-box, .secondary.svelte-10f983z {
    background-color: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    padding: 20px !important;
    margin-bottom: 24px !important;
}

/* Remove backgrounds from labels and markdown headers */
.gr-label, .gr-markdown {
    background-color: transparent !important;
}

label span, .gr-markdown h3 {
    background: transparent !important;
    color: #eee !important;
}

/* Unify input heights and alignment */
.gr-input, .gr-dropdown, .gr-radio, .gr-number, .gr-slider {
    background-color: rgba(0, 0, 0, 0.2) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
}

/* Ensure Row items align at the top */
.gr-row {
    align-items: flex-start !important;
    gap: 16px !important;
}

/* Premium Button Styling */
.gr-button-primary {
    background: linear-gradient(135deg, #ff4c00 0%, #ff8c00 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(255, 76, 0, 0.3) !important;
}

.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(255, 76, 0, 0.4) !important;
}

.gr-button-stop {
    background: linear-gradient(135deg, #ff2e2e 0%, #ff5e5e 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* Progress bar styling */
.gr-progress {
    background-color: rgba(255, 255, 255, 0.1) !important;
    border-radius: 4px !important;
}

/* Hide progress bar on specific containers */
.no-progress .progress-view, .no-progress .gr-progress-view {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* Tab styling */
.gr-tabs {
    border: none !important;
}

.gr-tab-item {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
}

.gr-tab-item.selected {
    border-bottom: 2px solid #ff4c00 !important;
    color: #ff4c00 !important;
}
"""

with gr.Blocks(title="Qwen3-TTS Easy Finetuning", css=css) as app:
    gr.Markdown("# 🎙️ Qwen3-TTS Easy Finetuning WebUI")
    gr.Markdown("![License](https://img.shields.io/github/license/mozi1924/Qwen3-TTS-EasyFinetuning?style=for-the-badge&color=blue)")
    gr.Markdown("[GitHub Repo](https://github.com/mozi1924/Qwen3-TTS-EasyFinetuning) | **Author:** [Mozi Arasaka](https://github.com/mozi1924)")
    gr.Markdown("## ⚠️ Disclaimer")
    gr.Markdown("By using this tool to fine-tune models, you agree that you will not use it for any illegal purposes or to infringe upon the rights of others. The author is not responsible for any direct or indirect consequences arising from the use of this tool, including but not limited to hardware damage or legal disputes.")
    
    build_info = get_build_info()
    if build_info:
        gr.Markdown(f"**Image Build Time:** `{build_info.get('build_time')}` | **Git Hash:** `{build_info.get('git_hash')}`")
    else:
        # If running locally via python src/webui.py or volume mounted, build_info.json won't exist in the container root
        gr.Markdown(f"**Running Mode:** `Local / Volume Mounted` (No image build metadata found)")
    
    gpus_list = get_gpus()
    default_gpu = gpus_list[0] if gpus_list else "cpu"
    
    with gr.Tabs() as main_tabs:
        with gr.Tab("1. Data Preparation", id="data"):
            gr.Markdown("Autonomously clean, split, transcribe, and ready your data.")
            
            with gr.Column(elem_classes="gr-group"):
                gr.Markdown("### Step 1: Audio Split and Silence Filter")
                gr.Markdown("Reads the raw `.wav` folder and extracts chunks by filtering out silences, then resamples into an `audio_24k` folder.")
                with gr.Row():
                    global_speaker_input = gr.Textbox(
                        label="Speaker Name / Dataset Name", 
                        value="my_speaker", 
                        info="Required: Unique name for storage",
                        scale=1
                    )
                    input_dir = gr.Dropdown(
                        label="Raw WAVs Directory", 
                        choices=get_raw_datasets(),
                        value="raw-dataset",
                        allow_custom_value=True,
                        info="Folder containing source .wav files",
                        scale=1
                    )
                    ref_audio = gr.Dropdown(
                        label="Reference Audio Path", 
                        choices=get_ref_audios(),
                        value="raw-dataset/ref.wav", 
                        allow_custom_value=True,
                        info="Optional: Resampled to 24k",
                        scale=1
                    )
                    num_threads = gr.Slider(
                        minimum=1, maximum=32, step=1, value=6,
                        label="Processing Threads",
                        info="Number of threads for audio splitting",
                        scale=1
                    )
                    skip_split = gr.Checkbox(
                        label="Skip Split, Only Resample",
                        value=False,
                        info="Best for already-trimmed and already-labeled short clips",
                        scale=1,
                    )
                
                with gr.Row():
                    step1_btn = gr.Button("▶️ Run Step 1: Audio Split & Ref Process", variant="primary", scale=4)
                    step1_refresh_btn = gr.Button("🔄 Refresh Paths", scale=1)
                step1_out = gr.Textbox(label="Step 1 Output", lines=1)

            gr.Markdown("<br>")
            
            with gr.Column(elem_classes="gr-group"):
                gr.Markdown("### Step 2: ASR Transcription & Cleaning")
                gr.Markdown("Transcribes the `audio_24k` folder with a selected ASR model, outputs cleanly to `tts_train.jsonl`.")
                with gr.Row():
                    asr_model = gr.Dropdown(
                        ["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"], 
                        label="ASR Model", 
                        value="Qwen/Qwen3-ASR-1.7B",
                        info="Select recognition model size",
                        scale=2
                    )
                    asr_source = gr.Radio(
                        ["HuggingFace", "ModelScope"], 
                        label="Download Source", 
                        value="HuggingFace",
                        info="Preferred hub for ASR model",
                        scale=1
                    )
                    gpu_asr = gr.Dropdown(
                        gpus_list, 
                        label="GPU Device", 
                        value=default_gpu,
                        info="Device for ASR processing",
                        scale=1
                    )
                
                step2_btn = gr.Button("▶️ Run Step 2: ASR Transcription", variant="primary")
                step2_out = gr.Textbox(label="Step 2 Output", lines=1)
                
        with gr.Tab("2. Training (Fine-tuning)", id="training"):
            gr.Markdown("Complete data tokenization and train the Qwen3-TTS model.")
            
            with gr.Column(elem_classes="gr-group"):
                gr.Markdown("### Step 0: Model Selection & Environment")
                with gr.Row():
                    with gr.Column(scale=2):
                        experiment_dropdown = gr.Dropdown(get_experiments(), label="Experiment Name", allow_custom_value=True, info="Select existing, or type below to create")
                        with gr.Row():
                            new_exp_name = gr.Textbox(show_label=False, placeholder="New Experiment Name...", scale=2)
                            exp_new_btn = gr.Button("➕ New", variant="secondary", scale=1)
                        exp_refresh_btn = gr.Button("🔄 Refresh")
                    with gr.Column(scale=2):
                        speaker_dropdown = gr.Dropdown(get_datasets(), label="Select Target Speaker Data", allow_custom_value=True, multiselect=True, info="Select one or more speakers for multi-speaker training")
                        spk_refresh_btn = gr.Button("🔄 Refresh Speakers")
                
                with gr.Row():
                    with gr.Column():    
                        init_model = gr.Dropdown(["Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"], label="Initial Model (Base)", value="Qwen/Qwen3-TTS-12Hz-0.6B-Base", allow_custom_value=True, info="Starting weights")
                        model_source = gr.Radio(["HuggingFace", "ModelScope"], label="Source", value="HuggingFace")
                    with gr.Column():
                        download_btn = gr.Button("⬇️ Check / Download Model", variant="secondary")
                        download_log = gr.Textbox(label="Download Status", lines=1)
                    
            gr.Markdown("---")
                    
            # Row 2: Tokenization Data Prep
            with gr.Column(elem_classes="gr-group"):
                gr.Markdown("### Step 3: Data Tokenization")
                gr.Markdown("Converts transcribed audio text entries to Audio Codes using Tokenizer. Required before training.")
                with gr.Row():
                    gpu_prep = gr.Dropdown(
                        gpus_list, 
                        label="GPU Device for Tokenization", 
                        value=default_gpu,
                        info="Used for encoding audio into tokens",
                        scale=1
                    )
                step3_btn = gr.Button("▶️ Tokenize Data", variant="primary")
                step3_out = gr.Textbox(label="Tokenization Logs", lines=1)
                
            gr.Markdown("---")
            
            # Row 3: Training Parameters
            with gr.Column(elem_classes="gr-group"):
                gr.Markdown("### Step 4: Final Training")
                with gr.Row():
                    preset_dropdown = gr.Dropdown(list(presets.keys()), label="Training Preset", value="0.6B Model", info="Optimized parameter sets")
                    gpu_train = gr.Dropdown(gpus_list, label="GPU Device for Training", value=default_gpu, info="Target GPU for SFT")
                    
                with gr.Accordion("Advanced Training Options", open=False):
                    with gr.Row():
                        t_lr = gr.Textbox(label="Learning Rate", value="1e-7", info="e.g. 1e-7 or 0.000002")
                        t_epochs = gr.Slider(minimum=1, maximum=100, step=1, value=2, label="Epochs")
                        t_batch = gr.Slider(minimum=1, maximum=16, step=1, value=2, label="Batch Size")
                        t_grad = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="Gradient Accumulation")
                    
                    with gr.Row():
                        t_speedup = gr.Checkbox(label="Use Experimental Training Method to Speed Up (Multi-core CPU)", value=False)
                        t_use_accelerator = gr.Checkbox(label="Use Accelerate", value=False)
                    with gr.Row():
                        t_resume = gr.Dropdown(get_checkpoints(include_specials=True), label="Resume From Checkpoint", value="latest", info="Select 'latest' to auto-resume, 'none' to restart, or a specific folder")
                        t_save_strategy = gr.Dropdown(["both", "epoch", "step"], label="Save Strategy", value="both", info="Checkpoint save strategy")
                    with gr.Row():
                        t_save_steps = gr.Slider(minimum=10, maximum=5000, step=10, value=200, label="Save Every N Steps")
                        t_keep_ckpt = gr.Slider(minimum=1, maximum=20, step=1, value=3, label="Keep Last N Checkpoints Per Type")
                        
                with gr.Row():
                    train_btn = gr.Button("🚀 Start Training", variant="primary", elem_classes="gr-button-primary")
                    stop_btn = gr.Button("🛑 Stop Training", variant="stop", elem_classes="gr-button-stop")
                
                # Update Start Training inputs
                train_btn_inputs = [experiment_dropdown, speaker_dropdown, init_model, model_source, t_batch, t_lr, t_epochs, t_grad, gpu_train, t_speedup, t_resume, t_save_strategy, t_save_steps, t_keep_ckpt, t_use_accelerator]
                
                with gr.Row():
                    with gr.Column(scale=3):
                        train_status = gr.Textbox(label="Process Status", lines=1)
                        with gr.Row():
                            tb_link_btn = gr.Button("📊 Jump to Tensorboard", variant="secondary")
                            tb_stop_btn = gr.Button("⏹️ Stop Tensorboard", variant="secondary")
                    with gr.Column(scale=7, elem_classes="no-progress"):
                        log_box = gr.Textbox(label="Live Training Logs (Streams automatically)", lines=10)
            
        with gr.Tab("3. Inference / Testing", id="inference"):
            gr.Markdown("Test your trained checkpoints.")
            with gr.Column(elem_classes="gr-group"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            ckpt_dropdown = gr.Dropdown([""] + get_checkpoints(include_specials=False), label="Select Checkpoint", value="", scale=4, info="Select a checkpoint to auto-load, or empty to unload")
                            ckpt_refresh_btn = gr.Button("🔄", scale=1)
                        
                        with gr.Row():
                            test_speaker = gr.Dropdown(label="Speaker Name", choices=[], allow_custom_value=True, info="Target voice ID")
                            test_language = gr.Dropdown(label="Language", choices=["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Italian", "Auto"], value="English", allow_custom_value=True)
                        
                        test_instruct = gr.Textbox(label="Optional Instruct", placeholder="e.g. Speak faster and with a high pitch", value="")
                        test_text = gr.Textbox(label="Text to Synthesize", value="Hello, this is a test from my custom voice.", lines=4)
                        gpu_infer = gr.Dropdown(gpus_list, label="GPU Device", value=default_gpu, info="Inference device")
                        
                        test_btn = gr.Button("Synthesize Audio", variant="primary", elem_classes="gr-button-primary")
                    
                    with gr.Column(scale=1):
                        audio_out = gr.Audio(label="Generated Audio", interactive=False)
                        inference_status = gr.Textbox(label="Inference Status", lines=1)
                        
                        gr.Markdown("---")
                        gr.Markdown("### Memory Management")
                        unload_btn = gr.Button("Unload Model from VRAM", variant="stop", elem_classes="gr-button-stop")
                        unload_status = gr.Textbox(label="Unload Status", lines=1)
            
    # ------ Handlers ------
    # Load config handler
    def refresh_exps():
        return gr.update(choices=get_experiments())
    route_hint = gr.HTML("<div style='opacity:0.75;font-size:0.9em;margin-top:-8px;margin-bottom:8px;'>Direct routes: <code>#data?exp=your_exp</code> / <code>#training?exp=your_exp</code> / <code>#inference?ckpt=exp/checkpoint-*</code></div>")
    exp_refresh_btn.click(fn=refresh_exps, outputs=[experiment_dropdown])
    
    # New Experiment Handler
    new_exp_outputs = [experiment_dropdown, new_exp_name, preset_dropdown, init_model, t_batch, t_lr, t_epochs, t_grad, speaker_dropdown, t_speedup, t_resume, t_save_strategy, t_save_steps, t_keep_ckpt, t_use_accelerator, train_status, t_resume]
    exp_new_btn.click(fn=lambda name: on_new_experiment(name, get_experiments), inputs=[new_exp_name], outputs=new_exp_outputs)
    
    spk_refresh_btn.click(fn=refresh_datasets, outputs=[speaker_dropdown])
    
    experiment_dropdown.change(
        fn=load_experiment_config, 
        inputs=[experiment_dropdown], 
        outputs=[preset_dropdown, init_model, t_batch, t_lr, t_epochs, t_grad, speaker_dropdown, t_speedup, t_resume, t_save_strategy, t_save_steps, t_keep_ckpt, t_use_accelerator, train_status, t_resume]
    )
    
    # Update training click handler
    train_btn.click(fn=start_training, inputs=train_btn_inputs, outputs=[train_status, log_box])
    
    # Step 1
    def refresh_step1_paths():
        return gr.update(choices=get_raw_datasets()), gr.update(choices=get_ref_audios())
    
    step1_refresh_btn.click(fn=refresh_step1_paths, outputs=[input_dir, ref_audio])

    def on_input_dir_change(dir_path):
        if not dir_path:
            return "raw-dataset/ref.wav"
        # If dir_path is 'raw-dataset/my_speaker', returns 'raw-dataset/my_speaker/ref.wav'
        # If dir_path is 'raw-dataset', returns 'raw-dataset/ref.wav'
        return os.path.join(dir_path, "ref.wav")

    input_dir.change(fn=on_input_dir_change, inputs=[input_dir], outputs=[ref_audio])

    step1_btn.click(
        fn=run_step_1,
        inputs=[input_dir, global_speaker_input, ref_audio, num_threads, skip_split],
        outputs=[step1_out]
    )

    # Step 2
    step2_btn.click(fn=run_step_2, inputs=[global_speaker_input, asr_model, asr_source, gpu_asr], outputs=[step2_out])

    
    # Step 3
    step3_btn.click(fn=run_step_3, inputs=[speaker_dropdown, experiment_dropdown, gpu_prep], outputs=[step3_out])
    
    # Utilities
    download_btn.click(fn=check_or_download_model, inputs=[init_model, model_source], outputs=[download_log])
    preset_dropdown.change(fn=apply_preset, inputs=[preset_dropdown, experiment_dropdown], outputs=[init_model, t_lr, t_epochs, t_batch, t_grad])
    # Also auto change preset when init model changes if it matches
    def auto_preset(model_val):
        if "1.7B" in model_val: return "1.7B Model"
        return "0.6B Model"
    init_model.change(fn=auto_preset, inputs=[init_model], outputs=[preset_dropdown])
    
    # Training
    stop_btn.click(fn=stop_training, outputs=[train_status])

    # Tensorboard Handlers
    tb_link_btn.click(
        fn=None, 
        inputs=[], 
        outputs=[], 
        js="""() => { 
            let url = window.location.origin;
            if (url.includes(':7860')) {
                url = url.replace(':7860', ':6006');
            } else if (window.location.hostname.includes('7860')) {
                url = window.location.protocol + '//' + window.location.hostname.replace('7860', '6006');
            } else {
                url = window.location.protocol + '//' + window.location.hostname + ':6006';
            }
            window.open(url, '_blank'); 
        }"""
    )
    tb_stop_btn.click(fn=stop_tensorboard, outputs=[train_status])
    
    # Inference
    ckpt_refresh_btn.click(fn=lambda: gr.update(choices=[""] + get_checkpoints(include_specials=False)), outputs=[ckpt_dropdown])

    def on_checkpoint_or_device_change(model_path, gpu_id):
        status, speakers, languages = load_model(model_path, gpu_id)
        speaker_value = speakers[0] if speakers else None
        language_value = languages[0] if languages else None
        return status, gr.update(choices=speakers, value=speaker_value), gr.update(choices=languages, value=language_value)

    ckpt_dropdown.change(
        fn=on_checkpoint_or_device_change,
        inputs=[ckpt_dropdown, gpu_infer],
        outputs=[inference_status, test_speaker, test_language]
    )
    gpu_infer.change(
        fn=on_checkpoint_or_device_change,
        inputs=[ckpt_dropdown, gpu_infer],
        outputs=[inference_status, test_speaker, test_language]
    )

    unload_btn.click(fn=unload_model, outputs=[unload_status])
    test_btn.click(
        fn=run_inference,
        inputs=[ckpt_dropdown, test_speaker, test_language, test_text, test_instruct, gpu_infer],
        outputs=[audio_out, inference_status]
    )

if __name__ == "__main__":
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
