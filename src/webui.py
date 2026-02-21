import gradio as gr
import os
import subprocess
import json
import time
import torch
import gc
import sys
import threading

# Ensure src in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import run_pipeline
from utils import get_model_path

# ----------------- Globals & Utilities -----------------
global_tts_model = None
global_tts_model_path = None
global_tts_device = None

def get_gpus():
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]

def get_datasets():
    if not os.path.exists("final-dataset"): return []
    return [d for d in os.listdir("final-dataset") if os.path.isdir(os.path.join("final-dataset", d))]

def get_checkpoints():
    if not os.path.exists("output"): return []
    ckpts = []
    for speaker in os.listdir("output"):
        sp_path = os.path.join("output", speaker)
        if os.path.isdir(sp_path):
            for ckpt in os.listdir(sp_path):
                if ckpt.startswith("checkpoint-"):
                    ckpts.append(os.path.join(sp_path, ckpt))
    return ckpts

def is_process_running(keyword):
    try:
        out = subprocess.check_output(["pgrep", "-f", keyword]).decode("utf-8").strip()
        return out != ""
    except: return False

def tee_process_output(process, log_filepath, append=False):
    mode = "a" if append else "w"
    with open(log_filepath, mode) as f:
        for line in iter(process.stdout.readline, ""):
            if not line: break
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()

# ----------------- Data Pipeline -----------------
def run_data_prep(input_dir, ref_audio, speaker_name, model_id, asr_source, gpu_id, progress=gr.Progress()):
    if not speaker_name.strip(): return "Please specify a Speaker Name.", gr.update()
    output_dir = os.path.join("final-dataset", speaker_name.strip())
    os.makedirs(output_dir, exist_ok=True)
    
    if asr_source == "HuggingFace":
        os.environ["USE_HF"] = "1"
    else:
        os.environ.pop("USE_HF", None)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id.replace("cuda:", "") if gpu_id != "cpu" else ""
    
    try:
        use_hf = (asr_source == "HuggingFace")
        resolved_model_id = get_model_path(model_id, use_hf)
        
        success, msg = run_pipeline(
            input_dir=input_dir,
            ref_audio=ref_audio,
            output_dir=output_dir,
            model_id=resolved_model_id,
            batch_size=16,
            progress=progress
        )
        return msg, gr.update(choices=get_datasets(), value=speaker_name.strip())
    except Exception as e:
        return f"Error: {e}", gr.update()

# ----------------- Training -----------------
training_process = None

def start_training(speaker_name, init_model, model_source, batch_size, lr, epochs, grad_acc, gpu_id):
    global training_process
    
    unload_model() # Force unload model before training memory clears
    
    if training_process is not None and training_process.poll() is None:
        return "Training is already running!"
        
    if not speaker_name:
        return "Please select a dataset/speaker from the dropdown."
    
    raw_jsonl = os.path.join("final-dataset", speaker_name, "tts_train.jsonl")
    if not os.path.exists(raw_jsonl):
        return f"JSONL file {raw_jsonl} not found. Please run data prep first."
        
    output_dir = os.path.join("output", speaker_name)
    os.makedirs(output_dir, exist_ok=True)
    
    train_jsonl = raw_jsonl.replace(".jsonl", "_with_codes.jsonl")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id.replace("cuda:", "") if gpu_id != "cpu" else ""
    env["PYTHONPATH"] = "src:" + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered Python output for real-time logs
    
    use_hf = (model_source == "HuggingFace")
    resolved_init_model = get_model_path(init_model, use_hf)
    resolved_tokenizer = get_model_path("Qwen/Qwen3-TTS-Tokenizer-12Hz", use_hf)
    
    prep_cmd = [
        "python", "src/prepare_data.py", 
        "--device", "cuda:0" if gpu_id != "cpu" else "cpu",
        "--tokenizer_model_path", resolved_tokenizer, 
        "--input_jsonl", raw_jsonl, 
        "--output_jsonl", train_jsonl
    ]
    
    log_path = "training_log.txt"
    with open(log_path, "w") as f:
        f.write("=== Starting run ===\n")

    if not os.path.exists(train_jsonl):
        print("Running prepare_data.py for audio codecs...", flush=True)
        try:
            prep_proc = subprocess.Popen(prep_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            tee_process_output(prep_proc, log_path, append=True)
            prep_proc.wait()
            if prep_proc.returncode != 0:
                return f"Error extracting codes. See docker logs."
        except Exception as e:
            return f"Error extracting codes: {e}"
    
    cmd = [
        "python", "src/sft_12hz.py",
        "--init_model_path", resolved_init_model,
        "--output_model_path", output_dir,
        "--train_jsonl", train_jsonl,
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--num_epochs", str(epochs),
        "--speaker_name", speaker_name,
        "--gradient_accumulation_steps", str(grad_acc),
        "--resume_from_checkpoint", "latest"
    ]
    
    if model_source == "ModelScope":
        env["VLLM_USE_MODELSCOPE"] = "true"
    else:
        env.pop("VLLM_USE_MODELSCOPE", None)
    
    print("Running sft_12hz.py...", flush=True)
    training_process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    # Thread to stream to console and file asynchronously
    t = threading.Thread(target=tee_process_output, args=(training_process, log_path, True))
    t.daemon = True
    t.start()
    
    if not is_process_running("tensorboard --logdir logs"):
        subprocess.Popen(["tensorboard", "--logdir", "logs", "--port", "6006", "--bind_all"])
        
    return f"Training started (PID: {training_process.pid})"

def stop_training():
    global training_process
    if training_process is not None and training_process.poll() is None:
        training_process.terminate()
        return "Training stopped."
    return "No training process running."

def read_logs():
    if os.path.exists("training_log.txt"):
        with open("training_log.txt", "r") as f:
            lines = f.readlines()
            
            status_summary = "Initializing / Downloading models..."
            for line in reversed(lines):
                if "Epoch" in line and "Step" in line and "Loss" in line:
                    status_summary = f"🚀 **Training Progress**: {line.strip()}"
                    break
                elif "Resumed from checkpoint" in line:
                    status_summary = f"🔄 **{line.strip()}**"
                    break
                elif "Loading checkpoint shards" in line or "Loading" in line:
                    status_summary = f"⏬ **{line.strip()}**"
                    break
                
            log_tail = "".join(lines[-30:])
            return status_summary, log_tail
    return "Idle", "No logs available."

# ----------------- Inference -----------------
def load_model(model_path, gpu_id):
    global global_tts_model, global_tts_model_path, global_tts_device
    
    if global_tts_model_path == model_path and global_tts_device == gpu_id and global_tts_model is not None:
        return "Model already loaded."
        
    unload_model()
    
    try:
        resolved_model_path = get_model_path(model_path, use_hf=False)
        print(f"Loading {resolved_model_path} on {gpu_id}...")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id.replace("cuda:", "") if gpu_id != "cpu" else ""
        os.environ["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]
        
        target_device = "cuda:0" if gpu_id != "cpu" else "cpu"
        
        # dynamic import to avoid overhead if UI doesn't use it
        from qwen_tts import Qwen3TTSModel
        
        global_tts_model = Qwen3TTSModel.from_pretrained(
            resolved_model_path,
            device_map=target_device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if target_device.startswith("cuda") else None,
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
        # format is usually output/speaker_name/checkpoint-...
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

# ----------------- UI -----------------
with gr.Blocks(title="Qwen3-TTS Easy Finetuning") as app:
    gr.Markdown("# 🎙️ Qwen3-TTS Easy Finetuning WebUI")
    
    gpus_list = get_gpus()
    default_gpu = gpus_list[0] if gpus_list else "cpu"
    
    with gr.Tabs():
        with gr.Tab("1. Data Preparation"):
            gr.Markdown("Auto split, transcribe (ASR), clean, and resample your dataset to 24kHz.")
            with gr.Row():
                with gr.Column():
                    speaker_name_input = gr.Textbox(label="Speaker Name (Required)", value="my_speaker", placeholder="e.g. crypto")
                    input_dir = gr.Textbox(label="Raw WAVs Directory", value="/workspace/raw-dataset", placeholder="Directory containing wav files")
                    ref_audio = gr.Textbox(label="Reference Audio Path (For TTS)", value="/workspace/raw-dataset/ref.wav", placeholder="Clear sounding audio clip (~3-10s)")
                
                with gr.Column():
                    asr_model = gr.Dropdown(["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"], label="ASR Model", value="Qwen/Qwen3-ASR-1.7B")
                    asr_source = gr.Radio(["ModelScope", "HuggingFace"], label="ASR Download Source", value="ModelScope")
                    gpu_asr = gr.Dropdown(gpus_list, label="GPU Device", value=default_gpu)
                    prep_btn = gr.Button("Start Data Processing", variant="primary")
                    prep_out = gr.Textbox(label="Processing Output", lines=5)
        
        with gr.Tab("2. Training (Fine-tuning)"):
            gr.Markdown("Finetune Qwen3-TTS model with your processed dataset.")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        dataset_dropdown = gr.Dropdown(get_datasets(), label="Select Dataset / Speaker", value=None, scale=3)
                        dataset_refresh_btn = gr.Button("🔄 Refresh", scale=1)
                        
                    preset_dropdown = gr.Dropdown(list(presets.keys()), label="Training Preset", value="0.6B Model")
                    init_model = gr.Dropdown(["Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"], label="Initial Model (Base)", value="Qwen/Qwen3-TTS-12Hz-0.6B-Base", allow_custom_value=True)
                    model_source = gr.Radio(["ModelScope", "HuggingFace"], label="Model Download Source", value="HuggingFace")
                    gpu_train = gr.Dropdown(gpus_list, label="GPU Device", value=default_gpu)
                    
                with gr.Column():
                    t_lr = gr.Number(label="Learning Rate", value=1e-7)
                    t_epochs = gr.Slider(minimum=1, maximum=100, step=1, value=2, label="Epochs")
                    t_batch = gr.Slider(minimum=1, maximum=16, step=1, value=2, label="Batch Size")
                    t_grad = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="Gradient Accumulation")
                    
            with gr.Row():
                train_btn = gr.Button("Start Training", variant="primary")
                stop_btn = gr.Button("Stop Training", variant="stop")
            
            with gr.Row():
                train_status = gr.Textbox(label="Process Status", lines=1)
                parsed_status = gr.Markdown("Idle")
                
            log_box = gr.Textbox(label="Live Training Logs", lines=10)
            
            # Auto refresh logs using gr.Timer
            with gr.Row():
                refresh_log_btn = gr.Button("Manual Refresh Logs")
                auto_refresh = gr.Checkbox(label="Auto Refresh Logs (Every 3s)", value=False)
                
            timer = gr.Timer(3, active=False)
            auto_refresh.change(lambda x: gr.Timer(active=x), auto_refresh, timer)
            timer.tick(fn=read_logs, outputs=[parsed_status, log_box])
            
        with gr.Tab("3. Inference / Testing"):
            gr.Markdown("Test your trained checkpoints. Model stays loaded in VRAM once selected.")
            with gr.Row():
                # Left Column: Inputs
                with gr.Column(scale=1):
                    with gr.Row():
                        ckpt_dropdown = gr.Dropdown(get_checkpoints(), label="Select Checkpoint", value=None, scale=4)
                        ckpt_refresh_btn = gr.Button("🔄", scale=1)
                        
                    test_speaker = gr.Textbox(label="Speaker Name (Auto-filled)", value="my_speaker")
                    test_text = gr.Textbox(label="Text to Synthesize", value="Hello, this is a test from my custom voice.", lines=4)
                    gpu_infer = gr.Dropdown(gpus_list, label="GPU Device", value=default_gpu)
                    
                    test_btn = gr.Button("Synthesize Audio", variant="primary")
                
                # Right Column: Outputs
                with gr.Column(scale=1):
                    audio_out = gr.Audio(label="Generated Audio", interactive=False)
                    inference_status = gr.Textbox(label="Inference Status", lines=1)
                    
                    gr.Markdown("---")
                    gr.Markdown("### Memory Management")
                    unload_btn = gr.Button("Unload Model from VRAM", variant="stop")
                    unload_status = gr.Textbox(label="Unload Status", lines=1)
            
    # ------ Handlers ------
    prep_btn.click(fn=run_data_prep, inputs=[input_dir, ref_audio, speaker_name_input, asr_model, asr_source, gpu_asr], outputs=[prep_out, dataset_dropdown])
    dataset_refresh_btn.click(fn=refresh_datasets, outputs=[dataset_dropdown])
    preset_dropdown.change(fn=apply_preset, inputs=[preset_dropdown], outputs=[init_model, t_lr, t_epochs, t_batch, t_grad])
    
    train_btn.click(fn=start_training, inputs=[dataset_dropdown, init_model, model_source, t_batch, t_lr, t_epochs, t_grad, gpu_train], outputs=[train_status])
    stop_btn.click(fn=stop_training, outputs=[train_status])
    refresh_log_btn.click(fn=read_logs, outputs=[parsed_status, log_box])
    
    ckpt_refresh_btn.click(fn=refresh_checkpoints, outputs=[ckpt_dropdown])
    ckpt_dropdown.change(fn=on_checkpoint_change, inputs=[ckpt_dropdown], outputs=[test_speaker])
    unload_btn.click(fn=unload_model, outputs=[unload_status])
    test_btn.click(fn=run_inference, inputs=[ckpt_dropdown, test_speaker, test_text, gpu_infer], outputs=[audio_out, inference_status])

if __name__ == "__main__":
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
