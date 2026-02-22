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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import torch
import gc
import sys
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
    dataset_path = os.path.abspath("final-dataset")
    if not os.path.exists(dataset_path): return []
    return [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

def get_checkpoints():
    output_path = os.path.abspath("output")
    if not os.path.exists(output_path): return []
    ckpts = []
    for exp in os.listdir(output_path):
        exp_dir = os.path.join(output_path, exp)
        if not os.path.isdir(exp_dir): continue
        for item in os.listdir(exp_dir):
            if "checkpoint-epoch" in item:
                ckpts.append(os.path.join(output_path, exp, item))
    return sorted(ckpts)

def get_model_path(model_id, use_hf=False, gr_progress=None):
    # Potential paths to check
    model_name = model_id.split("/")[-1]
    
    candidates = [
        os.path.join("/workspace/models", model_id),
        os.path.join("/workspace/models", model_name),
        os.path.join("/workspace/models", model_id.replace(".", "___")),
        os.path.join("/workspace/models", model_id.split("/")[0], model_name.replace(".", "___")),
    ]
    
    for path in candidates:
        if os.path.exists(path) and any(os.path.isfile(os.path.join(path, f)) for f in ["config.json", "model.safetensors", "pytorch_model.bin"]):
            return path
            
    # If not found, prepare for download
    local_path = os.path.join("/workspace/models", model_id)
    tqdm_class = (lambda **kwargs: GradioTqdm(gr_progress=gr_progress, **kwargs)) if gr_progress else None


    if use_hf:
        print(f"Downloading {model_id} from HuggingFace to {local_path} (Universal Format)...")
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_dir=local_path, local_dir_use_symlinks=False, tqdm_class=tqdm_class)
    else:
        print(f"Downloading {model_id} from ModelScope to {local_path}...")
        from modelscope import snapshot_download
        # ModelScope doesn't easily expose tqdm_class in snapshot_download, but we'll try basic notification
        if gr_progress: gr_progress(0.1, desc=f"Downloading {model_id} from ModelScope...")
        return snapshot_download(model_id, cache_dir="/workspace/models")



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
def run_step_1(input_dir, speaker_name, ref_audio, progress=gr.Progress()):
    if not speaker_name.strip(): 
        yield "Please specify a Speaker Name first."
        return
        
    output_dir = os.path.abspath(os.path.join("final-dataset", speaker_name.strip(), "audio_24k"))
    
    last_status = "Starting..."
    ref_path = os.path.abspath(ref_audio) if ref_audio else None
    
    for item in stream_isolated(internal_run_step_1, os.path.abspath(input_dir), output_dir, ref_path):
        if isinstance(item, dict):
            msg_type = item.get("type", "")
            if msg_type == "progress":
                progress(item.get("progress", 0), desc=item.get("desc", ""))
                last_status = f"Running: {item.get('desc', '')}"
            elif msg_type == "done":
                progress(1.0, desc="Done")
                yield f"Success: {item.get('msg', 'Completed')}"
                return
            elif msg_type == "error":
                progress(0, desc="Error")
                yield f"Error: {item.get('msg', 'Unknown Error')}"
                return
        elif isinstance(item, str):
            last_status = item
        yield last_status

# ----------------- Step 2: ASR Transcription -----------------
def run_step_2(speaker_name, asr_model, asr_source, gpu_id, progress=gr.Progress()):
    if not speaker_name.strip(): 
        yield "Please specify a Speaker Name first."
        return

    speaker_dir = os.path.abspath(os.path.join("final-dataset", speaker_name.strip()))
    input_dir = os.path.join(speaker_dir, "audio_24k")
    output_jsonl = os.path.join(speaker_dir, "tts_train.jsonl")
    
    # Auto-detect ref audio from previous step
    ref_24k = os.path.join(input_dir, "ref_24k.wav")
    ref_path = ref_24k if os.path.exists(ref_24k) else ""
    
    if not os.path.exists(input_dir):
        yield f"Error: Directory {input_dir} not found. Please run Step 1 first."
        return
        
    use_hf = (asr_source == "HuggingFace")
    resolved_model_id = get_model_path(asr_model, use_hf, gr_progress=progress)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id.replace("cuda:", "") if gpu_id != "cpu" else ""

    last_status = "Starting..."
    for item in stream_isolated(internal_run_step_2, input_dir, ref_path, output_jsonl, resolved_model_id, batch_size=16):
        if isinstance(item, dict):
            msg_type = item.get("type", "")
            if msg_type == "progress":
                progress(item.get("progress", 0), desc=item.get("desc", ""))
                last_status = f"Running: {item.get('desc', '')}"
            elif msg_type == "done":
                progress(1.0, desc="Done")
                yield f"Success: {item.get('msg', 'Completed')}"
                return
            elif msg_type == "error":
                progress(0, desc="Error")
                yield f"Error: {item.get('msg', 'Unknown Error')}"
                return
        elif isinstance(item, str):
            last_status = item
        yield last_status

# ----------------- Step 3: Tokenization -----------------
def run_step_3(speaker_name, gpu_id, progress=gr.Progress()):
    if not speaker_name.strip(): 
        yield "Please specify a Speaker Name first."
        return

    speaker_dir = os.path.abspath(os.path.join("final-dataset", speaker_name.strip()))
    input_jsonl = os.path.join(speaker_dir, "tts_train.jsonl")
    output_jsonl = os.path.join(speaker_dir, "tts_train_with_codes.jsonl")
    
    if not os.path.exists(input_jsonl):
        yield f"Error: File {input_jsonl} not found. Please run Data Prep Step 1 & 2 first."
        return
        
    resolved_tokenizer = get_model_path("Qwen/Qwen3-TTS-Tokenizer-12Hz", use_hf=False, gr_progress=progress)
    device = "cuda:0" if gpu_id != "cpu" else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id.replace("cuda:", "") if gpu_id != "cpu" else ""

    last_status = "Starting..."
    for item in stream_isolated(internal_run_prepare, device, resolved_tokenizer, input_jsonl, output_jsonl):
        if isinstance(item, dict):
            msg_type = item.get("type", "")
            if msg_type == "progress":
                progress(item.get("progress", 0), desc=item.get("desc", ""))
                last_status = f"Running: {item.get('desc', '')}"
            elif msg_type == "done":
                progress(1.0, desc="Done")
                yield f"Success: {item.get('msg', 'Completed')}"
                return
            elif msg_type == "error":
                progress(0, desc="Error")
                yield f"Error: {item.get('msg', 'Unknown Error')}"
                return
        elif isinstance(item, str):
            last_status = item
        yield last_status


# ----------------- Download Model -----------------
def check_or_download_model(init_model, model_source, progress=gr.Progress()):
    use_hf = (model_source == "HuggingFace")
    progress(0, desc="Preparing download...")
    try:
        resolved_init_model = get_model_path(init_model, use_hf, gr_progress=progress)
        return f"Model is ready at: {resolved_init_model}"
    except Exception as e:
        return f"Error downloading model: {e}"


def check_tb():
    def is_process_running(process_name):
        try:
            output = subprocess.check_output(["pgrep", "-f", process_name]).decode().strip()
            return bool(output)
        except subprocess.CalledProcessError:
            return False

    if not is_process_running("tensorboard --logdir logs"):
        subprocess.Popen(["tensorboard", "--logdir", "logs", "--port", "6006", "--bind_all"])

# ----------------- Training -----------------
def start_training(experiment_name, speaker_name, init_model, model_source, batch_size, lr, epochs, grad_acc, gpu_id, progress=gr.Progress()):
    global global_training_stop_event
    
    unload_model() # Force unload model before training memory clears
        
    if not experiment_name or not speaker_name:
        yield "Error: Please select or type an Experiment Name / Speaker Name.", ""
        return
    
    train_jsonl = os.path.abspath(os.path.join("final-dataset", speaker_name, "tts_train_with_codes.jsonl"))
    if not os.path.exists(train_jsonl):
        yield f"Error: JSONL file {train_jsonl} not found. Please run tokenization (Step 1 -> 2 -> 3) first.", ""
        return
        
    output_dir = os.path.abspath(os.path.join("output", experiment_name))
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-save Configuration
    config_path = os.path.join(output_dir, "training_config.json")
    config_data = {
        "speaker_name": speaker_name,
        "init_model": init_model,
        "model_source": model_source,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "grad_acc": grad_acc
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)
        
    use_hf = (model_source == "HuggingFace")
    resolved_init_model = get_model_path(init_model, use_hf, gr_progress=progress)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id.replace("cuda:", "") if gpu_id != "cpu" else ""

    check_tb()
    print(f"Starting in-process training on {gpu_id}...")
    
    log_history = []
    last_status = "Starting..."
    
    gen = stream_isolated(
        internal_run_train,
        experiment_name=experiment_name,
        init_model_path=resolved_init_model,
        output_model_path=output_dir,
        train_jsonl=train_jsonl,
        speaker_name=speaker_name,
        batch_size=batch_size,
        lr=lr,
        num_epochs=epochs,
        gradient_accumulation_steps=grad_acc,
        resume_from_checkpoint="latest"
    )

    try:
        for item in gen:
            if isinstance(item, dict):
                msg_type = item.get("type", "")
                if msg_type == "progress":
                    progress(item.get("progress", 0), desc=item.get("desc", ""))
                    last_status = f"Running: {item.get('desc', '')}"
                elif msg_type == "train_progress":
                    epoch = item.get("epoch", 0)
                    step = item.get("step", 0)
                    loss = item.get("loss", 0.0)
                    if isinstance(step, int):
                        desc_str = f"Epoch {epoch} | Step {step} | Loss: {loss:.4f}"
                    else:
                        desc_str = f"Epoch {epoch} | {step}"
                    progress(0.5, desc=desc_str)
                    last_status = desc_str
                    log_history.append(desc_str)
                elif msg_type == "done":
                    progress(1.0, desc="Done")
                    last_status = f"Success: {item.get('msg', 'Completed')}"
                    log_history.append(last_status)
                    yield last_status, "\n".join(log_history[-30:])
                    return
                elif msg_type == "error":
                    progress(0, desc="Error")
                    last_status = f"Error: {item.get('msg', 'Unknown Error')}"
                    log_history.append(last_status)
                    yield last_status, "\n".join(log_history[-30:])
                    return
            elif isinstance(item, str):
                log_history.append(item)
                last_status = item
            yield last_status, "\n".join(log_history[-30:])
    except Exception as e:
        yield f"Error: Unhandled exception {str(e)}", "\n".join(log_history[-30:])


def stop_training():
    global global_training_stop_event
    if global_training_stop_event is not None:
        global_training_stop_event.set()
        return "Notified training process to abort safely!"
    return "No training process currently running."


def load_experiment_config(experiment_name):
    # Returns: preset_dropdown, init_model, t_batch, t_lr, t_epochs, t_grad, speaker_dropdown, status_text
    config_path = os.path.join("output", experiment_name, "training_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            
            # Detect preset based on model size for UI convenience
            preset = "0.6B Model"
            if "1.7B" in data.get("init_model", ""):
                preset = "1.7B Model"
                
            return (
                preset,
                data.get("init_model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
                data.get("batch_size", 2),
                data.get("lr", 1e-7),
                data.get("epochs", 2),
                data.get("grad_acc", 4),
                data.get("speaker_name", ""),
                f"Loaded configuration for experiment '{experiment_name}'"
            )
        except Exception as e:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"Failed to load config: {e}"
            
    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "New experiment / No config found."

# ----------------- Inference -----------------
def load_model(model_path, gpu_id):
    global global_tts_model, global_tts_model_path, global_tts_device
    
    if global_tts_model_path == model_path and global_tts_device == gpu_id and global_tts_model is not None:
        return "Model already loaded."
        
    unload_model()
    
    try:
        resolved_model_path = get_model_path(model_path, use_hf=False)
        print(f"Loading {resolved_model_path} on {gpu_id}...")
        
        target_device = "cuda:0" if gpu_id != "cpu" else "cpu"
        # We also set the default logic when initializing model so that it overrides what the script is running on
        
        from qwen_tts import Qwen3TTSModel
        
        global_tts_model = Qwen3TTSModel.from_pretrained(
            resolved_model_path,
            device_map=gpu_id, # Actually pass the accurate gpu id e.g., cuda:1
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if "cuda" in gpu_id else None,
        )
        global_tts_model_path = model_path
        global_tts_device = gpu_id
        return f"Loaded {model_path} successfully."
    except Exception as e:
        return f"Failed to load model: {e}"

def unload_model():
    global global_tts_model, global_tts_model_path, global_tts_device
    if global_tts_model is not None:
        del global_tts_model
        global_tts_model = None
        global_tts_model_path = None
        global_tts_device = None
        gc.collect()
        torch.cuda.empty_cache()
        return "Model unloaded and VRAM cleared."
    return "No model was loaded."

def run_inference(model_path, speaker, text, gpu_id, progress=gr.Progress()):
    global global_tts_model, global_tts_model_path
    if not model_path:
        return None, "Please select a model checkpoint."
        
    if global_tts_model_path != model_path or global_tts_model is None:
        progress(0.1, desc="Loading Model into VRAM...")
        load_msg = load_model(model_path, gpu_id)
        if "Failed" in load_msg:
            return None, load_msg
            
    try:
        progress(0.5, desc="Synthesizing audio...")
        import soundfile as sf
        wavs, sr = global_tts_model.generate_custom_voice(
            text=text,
            speaker=speaker,
        )
        out_path = "webui_output.wav"
        sf.write(out_path, wavs[0], sr)
        progress(1.0, desc="Done!")
        return out_path, "Inference successful."
    except Exception as e:
        return None, f"Inference error: {e}"

# UI Event Callbacks
def on_checkpoint_change(ckpt_path):
    if ckpt_path and "output/" in ckpt_path:
        dirs = ckpt_path.replace("\\", "/").split("/")
        if len(dirs) >= 3:
            return dirs[-2]
    return "my_speaker"

def refresh_checkpoints():
    return gr.update(choices=get_checkpoints())

def refresh_datasets():
    return gr.update(choices=get_datasets())

presets = {
    "0.6B Model": { "init_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "lr": 1e-7, "epochs": 2, "batch_size": 2, "grad_acc": 4 },
    "1.7B Model": { "init_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "lr": 2e-6, "epochs": 3, "batch_size": 2, "grad_acc": 1 }
}

def apply_preset(preset_name):
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
    
    gpus_list = get_gpus()
    default_gpu = gpus_list[0] if gpus_list else "cpu"
    
    with gr.Tabs():
        with gr.Tab("1. Data Preparation"):
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
                    input_dir = gr.Textbox(
                        label="Raw WAVs Directory", 
                        value="/workspace/raw-dataset",
                        info="Folder containing source .wav files",
                        scale=1
                    )
                    ref_audio = gr.Textbox(
                        label="Reference Audio Path", 
                        value="/workspace/raw-dataset/ref.wav", 
                        info="Optional: Resampled to 24k",
                        scale=1
                    )
                
                step1_btn = gr.Button("▶️ Run Step 1: Audio Split & Ref Process", variant="primary")
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
                        ["ModelScope", "HuggingFace"], 
                        label="Download Source", 
                        value="ModelScope",
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
                
        with gr.Tab("2. Training (Fine-tuning)"):
            gr.Markdown("Complete data tokenization and train the Qwen3-TTS model.")
            
            with gr.Column(elem_classes="gr-group"):
                gr.Markdown("### Step 0: Model Selection & Environment")
                with gr.Row():
                    with gr.Column(scale=2):
                        experiment_dropdown = gr.Dropdown(get_experiments(), label="Experiment Name", allow_custom_value=True, info="Select existing or type a new one")
                        exp_refresh_btn = gr.Button("🔄 Refresh / Load Config", size="sm")
                    with gr.Column(scale=2):
                        speaker_dropdown = gr.Dropdown(get_datasets(), label="Select Target Speaker Data", allow_custom_value=True, info="Source dataset for fine-tuning")
                        spk_refresh_btn = gr.Button("🔄 Refresh Speakers", size="sm")
                
                with gr.Row():
                    with gr.Column():    
                        init_model = gr.Dropdown(["Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"], label="Initial Model (Base)", value="Qwen/Qwen3-TTS-12Hz-0.6B-Base", allow_custom_value=True, info="Starting weights")
                        model_source = gr.Radio(["ModelScope", "HuggingFace"], label="Source", value="HuggingFace")
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
                        t_lr = gr.Number(label="Learning Rate", value=1e-7)
                        t_epochs = gr.Slider(minimum=1, maximum=100, step=1, value=2, label="Epochs")
                        t_batch = gr.Slider(minimum=1, maximum=16, step=1, value=2, label="Batch Size")
                        t_grad = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="Gradient Accumulation")
                        
                with gr.Row():
                    train_btn = gr.Button("🚀 Start Training", variant="primary", elem_classes="gr-button-primary")
                    stop_btn = gr.Button("🛑 Stop Training", variant="stop", elem_classes="gr-button-stop")
                
                with gr.Row():
                    train_status = gr.Textbox(label="Process Status", lines=1)
                    log_box = gr.Textbox(label="Live Training Logs (Streams automatically)", lines=10)
            
        with gr.Tab("3. Inference / Testing"):
            gr.Markdown("Test your trained checkpoints.")
            with gr.Column(elem_classes="gr-group"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            ckpt_dropdown = gr.Dropdown(get_checkpoints(), label="Select Checkpoint", value=None, scale=4, info="Select a .pt or .safetensors checkpoint")
                            ckpt_refresh_btn = gr.Button("🔄", scale=1)
                            
                        test_speaker = gr.Textbox(label="Speaker Name (Auto-filled)", value="my_speaker", info="Target voice ID")
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
    exp_refresh_btn.click(fn=refresh_exps, outputs=[experiment_dropdown])
    spk_refresh_btn.click(fn=refresh_datasets, outputs=[speaker_dropdown])
    
    experiment_dropdown.change(
        fn=load_experiment_config, 
        inputs=[experiment_dropdown], 
        outputs=[preset_dropdown, init_model, t_batch, t_lr, t_epochs, t_grad, speaker_dropdown, train_status]
    )
    
    # Step 1
    step1_btn.click(fn=run_step_1, inputs=[input_dir, global_speaker_input, ref_audio], outputs=[step1_out])
    
    # Step 2
    step2_btn.click(fn=run_step_2, inputs=[global_speaker_input, asr_model, asr_source, gpu_asr], outputs=[step2_out])

    
    # Step 3
    step3_btn.click(fn=run_step_3, inputs=[speaker_dropdown, gpu_prep], outputs=[step3_out])
    
    # Utilities
    download_btn.click(fn=check_or_download_model, inputs=[init_model, model_source], outputs=[download_log])
    preset_dropdown.change(fn=apply_preset, inputs=[preset_dropdown], outputs=[init_model, t_lr, t_epochs, t_batch, t_grad])
    # Also auto change preset when init model changes if it matches
    def auto_preset(model_val):
        if "1.7B" in model_val: return "1.7B Model"
        return "0.6B Model"
    init_model.change(fn=auto_preset, inputs=[init_model], outputs=[preset_dropdown])
    
    # Training
    train_btn.click(
        fn=start_training, 
        inputs=[experiment_dropdown, speaker_dropdown, init_model, model_source, t_batch, t_lr, t_epochs, t_grad, gpu_train], 
        outputs=[train_status, log_box]
    )
    stop_btn.click(fn=stop_training, outputs=[train_status])
    
    # Inference
    ckpt_refresh_btn.click(fn=refresh_checkpoints, outputs=[ckpt_dropdown])
    ckpt_dropdown.change(fn=on_checkpoint_change, inputs=[ckpt_dropdown], outputs=[test_speaker])
    unload_btn.click(fn=unload_model, outputs=[unload_status])
    test_btn.click(fn=run_inference, inputs=[ckpt_dropdown, test_speaker, test_text, gpu_infer], outputs=[audio_out, inference_status])

if __name__ == "__main__":
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
