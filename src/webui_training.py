import json
import os
import threading
import time

import gradio as gr

from utils import resolve_path


def checkpoint_sort_key(output_path, exp_name, checkpoint_name):
    checkpoint_dir = os.path.join(output_path, exp_name, checkpoint_name)
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    global_step = -1
    epoch = -1
    if os.path.exists(trainer_state_path):
        try:
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                trainer_state = json.load(f)
            global_step = int(trainer_state.get("global_step", -1))
            epoch = int(trainer_state.get("epoch", -1))
        except Exception:
            pass
    return (global_step, epoch, checkpoint_name)


def get_checkpoints(experiment_name=None, include_specials=True):
    output_path = resolve_path("output")
    ckpts = ["latest", "none"] if include_specials else []

    if not os.path.exists(output_path):
        return ckpts

    exps = [experiment_name] if experiment_name else os.listdir(output_path)
    found_ckpts = []
    for exp in exps:
        exp_dir = os.path.join(output_path, exp)
        if not os.path.isdir(exp_dir):
            continue
        for item in os.listdir(exp_dir):
            if item.startswith("checkpoint-step-") or item.startswith("checkpoint-epoch-"):
                found_ckpts.append((exp, item))

    found_ckpts.sort(key=lambda x: checkpoint_sort_key(output_path, x[0], x[1]), reverse=True)
    return ckpts + [os.path.join(exp, item) for exp, item in found_ckpts]


def normalize_speaker_name(speaker_name):
    if isinstance(speaker_name, list):
        return ",".join(s.strip() for s in speaker_name if s and s.strip())
    return speaker_name.strip() if speaker_name else ""


def normalize_resume_checkpoint(resume_from_checkpoint):
    if resume_from_checkpoint == "none":
        return None
    if resume_from_checkpoint and resume_from_checkpoint != "latest" and not os.path.isabs(resume_from_checkpoint):
        return resolve_path(os.path.join("output", resume_from_checkpoint))
    return resume_from_checkpoint


def save_training_config(output_dir, config_data):
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)


def build_training_kwargs(
    experiment_name,
    speaker_name_str,
    resolved_init_model,
    output_dir,
    train_jsonl,
    batch_size,
    lr,
    epochs,
    grad_acc,
    final_resume,
    use_experimental_speedup,
    save_strategy,
    save_steps,
    keep_last_n_checkpoints,
    use_accelerator,
):
    return {
        "experiment_name": experiment_name,
        "init_model_path": resolved_init_model,
        "output_model_path": output_dir,
        "train_jsonl": train_jsonl,
        "speaker_name": speaker_name_str,
        "batch_size": batch_size,
        "lr": float(lr) if isinstance(lr, str) else lr,
        "num_epochs": epochs,
        "gradient_accumulation_steps": grad_acc,
        "resume_from_checkpoint": final_resume,
        "use_experimental_speedup": use_experimental_speedup,
        "save_strategy": save_strategy,
        "save_steps": save_steps,
        "keep_last_n_checkpoints": keep_last_n_checkpoints,
        "use_accelerator": use_accelerator,
    }


def append_log(log_history, message, limit=30):
    log_history.append(message)
    return "\n".join(log_history[-limit:])


def format_training_progress(item, total_epochs):
    epoch = item.get("epoch", 0)
    step = item.get("step", 0)
    loss = item.get("loss", 0.0)
    steps_in_epoch = item.get("steps_in_epoch")
    global_step = item.get("global_step")
    epoch_progress = item.get("epoch_progress")

    if isinstance(epoch_progress, (int, float)):
        current_progress = min(0.999, (epoch + float(epoch_progress)) / max(total_epochs, 1))
    else:
        current_progress = epoch / max(total_epochs, 1)

    step_prefix = f"Global Step {global_step} | " if isinstance(global_step, int) and global_step >= 0 else ""
    if isinstance(step, int):
        if isinstance(steps_in_epoch, int) and steps_in_epoch > 0:
            desc_str = f"Epoch {epoch + 1}/{total_epochs} | {step_prefix}Step {step + 1}/{steps_in_epoch} | Loss: {loss:.4f}"
        else:
            desc_str = f"Epoch {epoch + 1}/{total_epochs} | {step_prefix}Step {step + 1} | Loss: {loss:.4f}"
    else:
        desc_str = f"Epoch {epoch + 1}/{total_epochs} | {step_prefix}{step}"

    return max(0.0, min(current_progress, 0.999)), desc_str


def handle_training_message(item, progress, total_epochs, last_status, log_history):
    msg_type = item.get("type", "")
    if msg_type == "progress":
        progress(item.get("progress", 0), desc=item.get("desc", ""))
        last_status = f"Running: {item.get('desc', '')}"
        return last_status, append_log(log_history, last_status), False
    if msg_type in {"train_progress", "epoch_start"}:
        current_progress, desc_str = format_training_progress(item, total_epochs)
        progress(current_progress, desc=desc_str)
        return gr.update(), append_log(log_history, desc_str), False
    if msg_type == "done":
        progress(1.0, desc="Done")
        last_status = f"Success: {item.get('msg', 'Completed')}"
        return last_status, append_log(log_history, last_status), True
    if msg_type == "error":
        progress(0, desc="Error")
        last_status = f"Error: {item.get('msg', 'Unknown Error')}"
        return last_status, append_log(log_history, last_status), True
    return last_status, "\n".join(log_history[-30:]), False


def stream_worker_updates(stream, progress, success_prefix="Success"):
    last_status = "Starting..."
    for item in stream:
        if isinstance(item, dict):
            msg_type = item.get("type", "")
            if msg_type == "progress":
                progress(item.get("progress", 0), desc=item.get("desc", ""))
                last_status = f"Running: {item.get('desc', '')}"
            elif msg_type == "done":
                progress(1.0, desc="Done")
                yield f"{success_prefix}: {item.get('msg', 'Completed')}"
                return
            elif msg_type == "error":
                progress(0, desc="Error")
                yield f"Error: {item.get('msg', 'Unknown Error')}"
                return
        elif isinstance(item, str):
            last_status = item
        yield last_status


def get_deeplink_state(request: gr.Request):
    query = getattr(request, "query_params", {}) or {}
    return {
        "exp": query.get("exp", ""),
        "ckpt": query.get("ckpt", ""),
        "tab": query.get("tab", ""),
    }


def load_experiment_config(experiment_name):
    config_path = os.path.join("output", experiment_name, "training_config.json")
    checkpoint_choices = gr.update(choices=get_checkpoints(experiment_name=experiment_name, include_specials=True))
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            preset = "Latest Config"
            return (
                preset,
                data.get("init_model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
                data.get("batch_size", 2),
                data.get("lr", 1e-7),
                data.get("epochs", 2),
                data.get("grad_acc", 4),
                data.get("speaker_name", "").split(",") if data.get("speaker_name") else [],
                data.get("use_experimental_speedup", False),
                data.get("resume_from_checkpoint", "latest"),
                data.get("save_strategy", "both"),
                data.get("save_steps", 200),
                data.get("keep_last_n_checkpoints", 3),
                data.get("use_accelerator", False),
                f"Loaded configuration for experiment '{experiment_name}'",
                checkpoint_choices,
            )
        except Exception as e:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"Failed to load config: {e}", checkpoint_choices
            )

    return (
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "New experiment / No config found.", checkpoint_choices
    )


def on_new_experiment(name, get_experiments_fn):
    if not name or not name.strip():
        return [gr.update()] * 14 + ["Error: Experiment name cannot be empty.", gr.update()]

    name = name.strip()
    output_dir = resolve_path(os.path.join("output", name))

    if os.path.exists(output_dir):
        res = list(load_experiment_config(name))
        res[-2] = f"Experiment '{name}' already exists. Switched to it and loaded configuration."
        return [gr.update(choices=get_experiments_fn(), value=name), gr.update(value="")] + res

    try:
        os.makedirs(output_dir, exist_ok=True)
        return [
            gr.update(choices=get_experiments_fn(), value=name),
            gr.update(value=""),
            "0.6B Model",
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            2,
            "1e-7",
            2,
            4,
            [],
            False,
            "latest",
            "both",
            200,
            3,
            False,
            f"Successfully created new experiment: {name}",
            gr.update(choices=get_checkpoints(experiment_name=name, include_specials=True), value="latest"),
        ]
    except Exception as e:
        return [gr.update()] * 15 + [f"Error creating experiment folder: {e}", gr.update()]


def run_with_polling(fn, progress, progress_start=0.02, progress_end=0.95, desc_prefix="Downloading"):
    result = {"value": None, "error": None}

    def worker():
        try:
            result["value"] = fn()
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    last_time = time.time()
    last_size = 0
    target_dir = None
    try:
        maybe_path = getattr(fn, "target_dir", None)
        if maybe_path:
            target_dir = maybe_path
    except Exception:
        target_dir = None

    while thread.is_alive():
        current_time = time.time()
        if target_dir and os.path.exists(target_dir):
            total_size = 0
            for root, _, files in os.walk(target_dir):
                for name in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        pass
            delta_t = max(current_time - last_time, 1e-6)
            speed = max(total_size - last_size, 0) / delta_t
            downloaded_mb = total_size / (1024 ** 2)
            speed_mb = speed / (1024 ** 2)
            progress(progress_start, desc=f"{desc_prefix}... {downloaded_mb:.1f} MB | {speed_mb:.2f} MB/s")
            last_time = current_time
            last_size = total_size
        else:
            progress(progress_start, desc=f"{desc_prefix}...")
        time.sleep(0.3)

    thread.join()
    if result["error"] is not None:
        raise result["error"]
    progress(progress_end, desc=f"{desc_prefix} complete")
    return result["value"]
