# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import gc
import io
import json
import os
import random
import shutil
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None


target_speaker_embedding = None
speaker_embeddings = {}  # Dict[speaker_id_str, Tensor] for multi-speaker


# Setup args manually for dataset processing
class DummyArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def format_train_progress(epoch, step, loss, steps_in_epoch=None, global_step=None, epoch_progress=None):
    return {
        "type": "train_progress",
        "epoch": epoch,
        "step": step,
        "loss": float(loss),
        "steps_in_epoch": steps_in_epoch,
        "global_step": global_step,
        "epoch_progress": epoch_progress,
    }


def plot_spectrogram_to_numpy(spectrogram):
    """
    spectrogram: (time, freq) or (freq, time)
    Returns: numpy array HWC
    """
    if spectrogram.shape[0] > spectrogram.shape[1] and spectrogram.shape[1] == 128:
        # Likely (time, freq), transpose to (freq, time)
        spectrogram = spectrogram.T

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    from PIL import Image

    image = Image.open(buf).convert("RGB")
    return np.array(image)


def get_model_dtype(module):
    return next(module.parameters()).dtype


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_autocast_context(device, enabled=True):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def get_rng_state():
    state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def build_dataloader(dataset, batch_size, use_experimental_speedup=False, seed=42, epoch=0):
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch))

    common_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "collate_fn": dataset.collate_fn,
        "generator": generator,
    }

    if use_experimental_speedup:
        common_kwargs.update(
            {
                "num_workers": 4,
                "pin_memory": True,
                "prefetch_factor": 2,
            }
        )

    return DataLoader(**common_kwargs)


def find_latest_checkpoint(output_model_path):
    if not os.path.exists(output_model_path):
        return None

    candidates = []
    for name in os.listdir(output_model_path):
        full_path = os.path.join(output_model_path, name)
        if not os.path.isdir(full_path) or not name.startswith("checkpoint-"):
            continue

        trainer_state_file = os.path.join(full_path, "trainer_state.json")
        global_step = -1
        epoch = -1
        if os.path.exists(trainer_state_file):
            try:
                with open(trainer_state_file, "r", encoding="utf-8") as f:
                    trainer_state = json.load(f)
                global_step = int(trainer_state.get("global_step", -1))
                epoch = int(trainer_state.get("epoch", -1))
            except Exception:
                pass
        else:
            try:
                epoch = int(name.split("-")[-1])
                global_step = epoch
            except Exception:
                pass

        candidates.append((global_step, epoch, full_path))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[-1][2]


def save_trainer_state(checkpoint_dir, trainer_state):
    trainer_state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    with open(trainer_state_file, "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2, ensure_ascii=False)


def load_trainer_state(checkpoint_dir):
    trainer_state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.exists(trainer_state_file):
        return {}
    with open(trainer_state_file, "r", encoding="utf-8") as f:
        return json.load(f)


def export_inference_artifacts(
    checkpoint_dir,
    model_path,
    base_config,
    spk_id_map,
    model_to_export,
    current_speaker_embeddings,
    current_target_speaker_embedding,
    log_print,
):
    shutil.copytree(model_path, checkpoint_dir, dirs_exist_ok=True)

    output_config_file = os.path.join(checkpoint_dir, "config.json")
    config_dict = json.loads(json.dumps(base_config))
    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})
    talker_config["spk_id"] = spk_id_map
    talker_config["spk_is_dialect"] = {name: False for name in spk_id_map}
    config_dict["talker_config"] = talker_config

    with open(output_config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    state_dict = {k: v.detach().to("cpu") for k, v in model_to_export.state_dict().items()}

    drop_prefix = "speaker_encoder"
    keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
    for k in keys_to_drop:
        del state_dict[k]

    if not current_speaker_embeddings and current_target_speaker_embedding is None:
        log_print("Warning: no speaker embeddings were captured during training steps.")

    weight = state_dict["talker.model.codec_embedding.weight"]

    if current_speaker_embeddings:
        for spk_name, spk_idx in spk_id_map.items():
            if spk_name in current_speaker_embeddings:
                emb = current_speaker_embeddings[spk_name]
                state_dict["talker.model.codec_embedding.weight"][spk_idx] = emb.detach().to(weight.device).to(weight.dtype)
                log_print(f"Saved embedding for speaker '{spk_name}' at index {spk_idx}")
            else:
                log_print(f"Warning: no embedding captured for speaker '{spk_name}'")
    elif current_target_speaker_embedding is not None:
        state_dict["talker.model.codec_embedding.weight"][3000] = current_target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)

    save_path = os.path.join(checkpoint_dir, "model.safetensors")
    save_file(state_dict, save_path)


def save_checkpoint(
    checkpoint_dir,
    model_path,
    base_config,
    spk_id_map,
    model,
    optimizer,
    lr_scheduler,
    trainer_state,
    use_accelerator,
    accelerator,
    speaker_embeddings_state,
    target_speaker_embedding_state,
    log_print,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    save_trainer_state(checkpoint_dir, trainer_state)

    if use_accelerator:
        accel_state_dir = os.path.join(checkpoint_dir, "accelerate_state")
        os.makedirs(accel_state_dir, exist_ok=True)
        accelerator.save_state(accel_state_dir)
        unwrapped_model = accelerator.unwrap_model(model)
    else:
        manual_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if hasattr(lr_scheduler, "state_dict") else None,
            "speaker_embeddings": {k: v.detach().cpu() for k, v in speaker_embeddings_state.items()},
            "target_speaker_embedding": None
            if target_speaker_embedding_state is None
            else target_speaker_embedding_state.detach().cpu(),
            "rng_state": get_rng_state(),
        }
        torch.save(manual_state, os.path.join(checkpoint_dir, "training_state.pt"))
        unwrapped_model = model

    export_inference_artifacts(
        checkpoint_dir=checkpoint_dir,
        model_path=model_path,
        base_config=base_config,
        spk_id_map=spk_id_map,
        model_to_export=unwrapped_model,
        current_speaker_embeddings=speaker_embeddings_state,
        current_target_speaker_embedding=target_speaker_embedding_state,
        log_print=log_print,
    )


def normalize_save_strategy(save_strategy):
    strategy = str(save_strategy or "both").strip().lower()
    valid_strategies = {"step", "epoch", "both"}
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid save_strategy: {save_strategy}. Expected one of {sorted(valid_strategies)}")
    return strategy



def prune_old_checkpoints(output_model_path, keep_last_n, log_print):
    if keep_last_n is None or keep_last_n <= 0 or not os.path.exists(output_model_path):
        return

    checkpoints_by_type = {}
    for name in os.listdir(output_model_path):
        full_path = os.path.join(output_model_path, name)
        if not os.path.isdir(full_path) or not name.startswith("checkpoint-"):
            continue

        trainer_state = load_trainer_state(full_path)
        save_type = trainer_state.get("save_type", "unknown")
        global_step = int(trainer_state.get("global_step", -1))
        checkpoints_by_type.setdefault(save_type, []).append((global_step, full_path))

    for save_type, checkpoints in checkpoints_by_type.items():
        checkpoints.sort(key=lambda x: x[0])
        while len(checkpoints) > keep_last_n:
            _, old_path = checkpoints.pop(0)
            log_print(f"Pruning old {save_type} checkpoint: {old_path}")
            shutil.rmtree(old_path, ignore_errors=True)


def run_train(
    experiment_name,
    init_model_path,
    output_model_path,
    train_jsonl,
    speaker_name="speaker_test",
    batch_size=2,
    lr=1e-7,
    num_epochs=2,
    gradient_accumulation_steps=4,
    resume_from_checkpoint="latest",
    stop_event=None,
    use_experimental_speedup=False,
    save_steps=200,
    keep_last_n_checkpoints=3,
    seed=42,
    use_accelerator=True,
    save_strategy="both",
):
    global target_speaker_embedding, speaker_embeddings
    target_speaker_embedding = None
    speaker_embeddings = {}

    writer = None
    accelerator = None
    model = None
    optimizer = None
    lr_scheduler = None
    qwen3tts = None

    try:
        args = DummyArgs(
            init_model_path=init_model_path,
            output_model_path=output_model_path,
            train_jsonl=train_jsonl,
            speaker_name=speaker_name,
            batch_size=batch_size,
            lr=lr,
            num_epochs=num_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            resume_from_checkpoint=resume_from_checkpoint,
            use_experimental_speedup=use_experimental_speedup,
            save_steps=save_steps,
            keep_last_n_checkpoints=keep_last_n_checkpoints,
            seed=seed,
            use_accelerator=use_accelerator,
            save_strategy=normalize_save_strategy(save_strategy),
        )

        os.makedirs(args.output_model_path, exist_ok=True)

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_base = os.path.join(root_dir, "logs")
        os.makedirs(logs_base, exist_ok=True)
        log_dir = os.path.join(logs_base, experiment_name)

        if args.use_accelerator and Accelerator is None:
            yield {
                "type": "progress",
                "progress": 0.0,
                "desc": "accelerate 不可用，已自动切换到原生 PyTorch 训练模式。",
            }
            args.use_accelerator = False

        device = get_default_device()
        is_main_process = True

        if args.use_accelerator:
            accelerator = Accelerator(
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                mixed_precision="bf16",
            )
            device = accelerator.device
            is_main_process = accelerator.is_main_process

        MODEL_PATH = args.init_model_path
        qwen3tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        config = AutoConfig.from_pretrained(MODEL_PATH)
        with open(os.path.join(MODEL_PATH, "config.json"), "r", encoding="utf-8") as f:
            base_config = json.load(f)

        if isinstance(args.speaker_name, str):
            speaker_names = [s.strip() for s in args.speaker_name.split(",") if s.strip()]
        elif isinstance(args.speaker_name, list):
            speaker_names = args.speaker_name
        else:
            speaker_names = [str(args.speaker_name)]

        if not speaker_names:
            speaker_names = ["speaker_test"]

        spk_id_map = {name: 3000 + idx for idx, name in enumerate(speaker_names)}

        def log_print(msg):
            if accelerator is not None:
                accelerator.print(msg)
            else:
                print(msg)

        log_print(f"Multi-speaker mode: {len(speaker_names)} speakers")
        for name, sid in spk_id_map.items():
            log_print(f"  Speaker '{name}' -> spk_id {sid}")

        with open(args.train_jsonl, "r", encoding="utf-8") as f:
            train_data = [json.loads(line) for line in f]

        default_speaker = speaker_names[0]
        dataset = TTSDataset(train_data, qwen3tts.processor, config, default_speaker=default_speaker)

        model = qwen3tts.model
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        class DummyLRScheduler:
            def step(self):
                pass

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                return None

        lr_scheduler = DummyLRScheduler()

        if args.use_accelerator:
            model, optimizer = accelerator.prepare(model, optimizer)
        else:
            model = model.to(device)

        starting_epoch = 0
        resume_step = 0
        global_step = 0
        resumed = False

        if args.resume_from_checkpoint:
            resume_path = args.resume_from_checkpoint
            if isinstance(resume_path, str) and resume_path.lower() == "latest":
                resume_path = find_latest_checkpoint(args.output_model_path)

            if resume_path is not None and os.path.exists(resume_path):
                trainer_state = load_trainer_state(resume_path)
                if trainer_state:
                    starting_epoch = int(trainer_state.get("epoch", 0))
                    resume_step = int(trainer_state.get("step_in_epoch", 0))
                    global_step = int(trainer_state.get("global_step", 0))
                else:
                    try:
                        starting_epoch = int(os.path.basename(resume_path.rstrip("/\\")).split("-")[-1]) + 1
                    except Exception:
                        starting_epoch = 0
                    resume_step = 0
                    global_step = 0

                if args.use_accelerator:
                    accel_state_dir = os.path.join(resume_path, "accelerate_state")
                    if os.path.exists(accel_state_dir):
                        log_print(f"Resumed accelerator state from: {accel_state_dir}")
                        accelerator.load_state(accel_state_dir)
                else:
                    training_state_path = os.path.join(resume_path, "training_state.pt")
                    if os.path.exists(training_state_path):
                        log_print(f"Resumed native state from: {training_state_path}")
                        training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
                        model.load_state_dict(training_state["model"])
                        optimizer.load_state_dict(training_state["optimizer"])
                        if training_state.get("lr_scheduler") is not None:
                            lr_scheduler.load_state_dict(training_state["lr_scheduler"])
                        loaded_speaker_embeddings = training_state.get("speaker_embeddings", {})
                        speaker_embeddings = {k: v.cpu() for k, v in loaded_speaker_embeddings.items()}
                        loaded_target_embedding = training_state.get("target_speaker_embedding")
                        target_speaker_embedding = None if loaded_target_embedding is None else loaded_target_embedding.cpu()
                        set_rng_state(training_state.get("rng_state"))

                resumed = True
                log_print(
                    f"Resume training from checkpoint: {resume_path} | epoch={starting_epoch}, step_in_epoch={resume_step}, global_step={global_step}"
                )

        writer = SummaryWriter(log_dir=log_dir, purge_step=global_step if resumed else None)

        preview_dataloader = build_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            use_experimental_speedup=args.use_experimental_speedup,
            seed=args.seed,
            epoch=starting_epoch,
        )
        dataloader_length = len(preview_dataloader)

        log_print(f"Dataset size: {len(dataset)}")
        log_print(f"Dataloader length: {dataloader_length}")
        log_print(f"Global step starts at: {global_step}")
        log_print(f"Training mode: {'accelerate' if args.use_accelerator else 'native pytorch'}")
        log_print(f"Checkpoint save strategy: {args.save_strategy}")

        yield format_train_progress(
            starting_epoch,
            "Starting Training...",
            0.0,
            steps_in_epoch=dataloader_length,
            global_step=global_step,
            epoch_progress=0.0,
        )

        for epoch in range(starting_epoch, args.num_epochs):
            raw_dataloader = build_dataloader(
                dataset=dataset,
                batch_size=args.batch_size,
                use_experimental_speedup=args.use_experimental_speedup,
                seed=args.seed,
                epoch=epoch,
            )
            steps_in_epoch = len(raw_dataloader)
            train_dataloader = accelerator.prepare(raw_dataloader) if args.use_accelerator else raw_dataloader

            if is_main_process:
                yield {
                    "type": "epoch_start",
                    "epoch": epoch,
                    "step": 0,
                    "loss": 0.0,
                    "steps_in_epoch": steps_in_epoch,
                    "global_step": global_step,
                    "epoch_progress": 0.0,
                }

            model.train()
            optimizer.zero_grad()

            for step, batch in enumerate(train_dataloader):
                if stop_event and stop_event.is_set():
                    if is_main_process:
                        yield {"type": "progress", "progress": 1.0, "desc": "Training user-aborted."}
                    return

                if epoch == starting_epoch and step < resume_step:
                    if is_main_process and step % 100 == 0:
                        yield format_train_progress(
                            epoch,
                            f"Skipping {step}/{resume_step}...",
                            0.0,
                            steps_in_epoch=steps_in_epoch,
                            global_step=global_step,
                            epoch_progress=step / max(steps_in_epoch, 1),
                        )
                    continue

                if args.use_accelerator:
                    accumulation_context = accelerator.accumulate(model)
                    autocast_context = accelerator.autocast()
                else:
                    accumulation_context = nullcontext()
                    autocast_context = get_autocast_context(device, enabled=True)

                with accumulation_context:
                    model_device = accelerator.device if args.use_accelerator else device
                    unwrap_model = accelerator.unwrap_model(model) if args.use_accelerator else model
                    model_dtype = get_model_dtype(unwrap_model)

                    input_ids = batch["input_ids"].to(model_device)
                    codec_ids = batch["codec_ids"].to(model_device)
                    ref_mels_list = batch["ref_mels"]
                    text_embedding_mask = batch["text_embedding_mask"].to(model_device)
                    codec_embedding_mask = batch["codec_embedding_mask"].to(model_device)
                    attention_mask = batch["attention_mask"].to(model_device)
                    codec_0_labels = batch["codec_0_labels"].to(model_device)
                    codec_mask = batch["codec_mask"].to(model_device)

                    with autocast_context:
                        per_sample_embeddings = []
                        batch_speaker_ids = batch["speaker_ids"]
                        for b_idx, ref_mel in enumerate(ref_mels_list):
                            emb = unwrap_model.speaker_encoder(ref_mel.to(model_device).to(model_dtype)).detach()
                            per_sample_embeddings.append(emb)

                            sid = batch_speaker_ids[b_idx]
                            if sid not in speaker_embeddings:
                                speaker_embeddings[sid] = emb[0].cpu()
                                log_print(f"Captured embedding for speaker '{sid}'")

                        speaker_embedding = torch.cat(per_sample_embeddings, dim=0)

                        if target_speaker_embedding is None:
                            target_speaker_embedding = speaker_embedding.cpu()

                        input_text_ids = input_ids[:, :, 0]
                        input_codec_ids = input_ids[:, :, 1]

                        input_text_embedding = unwrap_model.talker.text_projection(
                            unwrap_model.talker.get_text_embeddings()(input_text_ids)
                        ) * text_embedding_mask
                        input_codec_embedding = (
                            unwrap_model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                        )
                        input_codec_embedding[:, 6, :] = speaker_embedding

                        input_embeddings = input_text_embedding + input_codec_embedding

                        for i in range(1, 16):
                            codec_i_embedding = unwrap_model.talker.code_predictor.get_input_embeddings()[i - 1](
                                codec_ids[:, :, i]
                            )
                            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                            input_embeddings = input_embeddings + codec_i_embedding

                        outputs = unwrap_model.talker(
                            inputs_embeds=input_embeddings[:, :-1, :],
                            attention_mask=attention_mask[:, :-1],
                            labels=codec_0_labels[:, 1:],
                            output_hidden_states=True,
                        )

                        hidden_states = outputs.hidden_states[0][-1]
                        talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                        talker_codec_ids = codec_ids[codec_mask]
                        _, sub_talker_loss = unwrap_model.talker.forward_sub_talker_finetune(
                            talker_codec_ids, talker_hidden_states
                        )
                        loss = outputs.loss + 0.3 * sub_talker_loss

                    if args.use_accelerator:
                        accelerator.backward(loss)
                        should_step = accelerator.sync_gradients
                    else:
                        scaled_loss = loss / args.gradient_accumulation_steps
                        scaled_loss.backward()
                        should_step = ((step + 1) % args.gradient_accumulation_steps == 0) or ((step + 1) == steps_in_epoch)

                    if should_step:
                        if args.use_accelerator:
                            accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                global_step += 1
                epoch_progress = (step + 1) / max(steps_in_epoch, 1)
                current_epoch_float = epoch + epoch_progress

                if is_main_process and step % 10 == 0:
                    yield format_train_progress(
                        epoch,
                        step,
                        loss.item(),
                        steps_in_epoch=steps_in_epoch,
                        global_step=global_step,
                        epoch_progress=epoch_progress,
                    )
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/talker_loss", outputs.loss.item(), global_step)
                    writer.add_scalar("train/sub_talker_loss", sub_talker_loss.item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("train/epoch", current_epoch_float, global_step)
                    writer.add_scalar("train/epoch_index", epoch, global_step)
                    writer.add_scalar("train/step_in_epoch", step, global_step)
                    writer.flush()

                if is_main_process and step % 500 == 0:
                    try:
                        mel_vis = plot_spectrogram_to_numpy(ref_mels_list[0][0].detach().cpu().float().numpy())
                        writer.add_image("train/ref_mel", mel_vis, global_step, dataformats="HWC")
                    except Exception as e:
                        log_print(f"Error logging Mel to Tensorboard: {e}")

                should_save_step_checkpoint = (
                    is_main_process
                    and args.save_strategy in {"step", "both"}
                    and args.save_steps is not None
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                )
                if should_save_step_checkpoint:
                    checkpoint_dir = os.path.join(args.output_model_path, f"checkpoint-step-{global_step}")
                    trainer_state = {
                        "epoch": epoch,
                        "step_in_epoch": step + 1,
                        "global_step": global_step,
                        "num_epochs": args.num_epochs,
                        "steps_in_epoch": steps_in_epoch,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "seed": args.seed,
                        "save_type": "step",
                    }
                    if args.use_accelerator:
                        accelerator.wait_for_everyone()
                    yield format_train_progress(
                        epoch,
                        f"Saving step checkpoint @ {global_step}",
                        loss.item(),
                        steps_in_epoch=steps_in_epoch,
                        global_step=global_step,
                        epoch_progress=epoch_progress,
                    )
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        model_path=MODEL_PATH,
                        base_config=base_config,
                        spk_id_map=spk_id_map,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        trainer_state=trainer_state,
                        use_accelerator=args.use_accelerator,
                        accelerator=accelerator,
                        speaker_embeddings_state=speaker_embeddings,
                        target_speaker_embedding_state=target_speaker_embedding,
                        log_print=log_print,
                    )
                    prune_old_checkpoints(args.output_model_path, args.keep_last_n_checkpoints, log_print)

            if args.use_accelerator:
                accelerator.wait_for_everyone()

            if is_main_process and args.save_strategy in {"epoch", "both"}:
                yield format_train_progress(
                    epoch,
                    "Saving Epoch Checkpoint",
                    0.0,
                    steps_in_epoch=steps_in_epoch,
                    global_step=global_step,
                    epoch_progress=1.0,
                )
                checkpoint_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
                trainer_state = {
                    "epoch": epoch + 1,
                    "step_in_epoch": 0,
                    "global_step": global_step,
                    "num_epochs": args.num_epochs,
                    "steps_in_epoch": steps_in_epoch,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "seed": args.seed,
                    "save_type": "epoch",
                }
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    model_path=MODEL_PATH,
                    base_config=base_config,
                    spk_id_map=spk_id_map,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    trainer_state=trainer_state,
                    use_accelerator=args.use_accelerator,
                    accelerator=accelerator,
                    speaker_embeddings_state=speaker_embeddings,
                    target_speaker_embedding_state=target_speaker_embedding,
                    log_print=log_print,
                )
                prune_old_checkpoints(args.output_model_path, args.keep_last_n_checkpoints, log_print)
                writer.add_scalar("train/epoch_completed", epoch + 1, global_step)
                writer.flush()

            resume_step = 0

        if is_main_process:
            writer.flush()
            yield {"type": "progress", "progress": 1.0, "desc": "Training completed."}
            yield {"type": "done", "msg": "Training saved successfully."}

    except Exception as e:
        yield {"type": "error", "msg": f"Unhandled exception in training: {str(e)}"}
    finally:
        if writer is not None:
            writer.close()
        if "qwen3tts" in locals() and qwen3tts is not None:
            del qwen3tts
        if "model" in locals() and model is not None:
            del model
        if accelerator is not None:
            accelerator.free_memory()
            del accelerator
        if "optimizer" in locals() and optimizer is not None:
            del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
