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
import json
import os
import shutil
import gc
import io
import numpy as np
import matplotlib.pyplot as plt

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None

# Setup args manually for dataset processing
class DummyArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def format_train_progress(epoch, step, loss):
    return {"type": "train_progress", "epoch": epoch, "step": step, "loss": float(loss)}

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
    stop_event=None
):
    global target_speaker_embedding
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
        )
        
        # Use project root logs directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_base = os.path.join(root_dir, "logs")
        os.makedirs(logs_base, exist_ok=True)
        
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps, 
            mixed_precision="bf16", 
            log_with="tensorboard",
            project_dir=logs_base,
        )
        accelerator.init_trackers(project_name=experiment_name)

        MODEL_PATH = args.init_model_path

        qwen3tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        config = AutoConfig.from_pretrained(MODEL_PATH)

        train_data = open(args.train_jsonl).readlines()
        train_data = [json.loads(line) for line in train_data]
        dataset = TTSDataset(train_data, qwen3tts.processor, config)
        train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

        optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)
        
        # Dummy lr_scheduler for compatibility with the provided snippet
        class DummyLRScheduler:
            def step(self):
                pass
        lr_scheduler = DummyLRScheduler()

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            qwen3tts.model, optimizer, train_dataloader, lr_scheduler
        )

        starting_epoch = 0
        resume_step = None # Initialize resume_step
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint.lower() == "latest":
                if os.path.exists(args.output_model_path):
                    dirs = [d for d in os.listdir(args.output_model_path) if d.startswith("accelerate-epoch-")]
                    if len(dirs) > 0:
                        dirs.sort(key=lambda x: int(x.split("-")[-1]))
                        args.resume_from_checkpoint = os.path.join(args.output_model_path, dirs[-1])
                    else:
                        args.resume_from_checkpoint = None
                else:
                    args.resume_from_checkpoint = None

            if args.resume_from_checkpoint is not None and os.path.exists(args.resume_from_checkpoint):
                accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                accelerator.load_state(args.resume_from_checkpoint)
                try:
                    starting_epoch = int(args.resume_from_checkpoint.split("-")[-1]) + 1
                except:
                    pass

        global_step = starting_epoch * len(train_dataloader)
        yield format_train_progress(0, "Starting Training...", 0.0)

        for epoch in range(starting_epoch, args.num_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                # Check for thread halt interrupt safely
                if stop_event and stop_event.is_set():
                    yield {"type": "progress", "progress": 1.0, "desc": "Training user-aborted."}
                    accelerator.end_training()
                    return
                    
                # Skip steps if resuming
                if resume_step is not None and step < resume_step:
                    if step % 100 == 0:
                        yield format_train_progress(epoch, f"Skipping {step}/{resume_step}...", 0.0)
                    continue
                
                with accelerator.accumulate(model):
                    input_ids = batch['input_ids'].to(model.device)
                    codec_ids = batch['codec_ids'].to(model.device)
                    ref_mels = batch['ref_mels'].to(model.device)
                    text_embedding_mask = batch['text_embedding_mask'].to(model.device)
                    codec_embedding_mask = batch['codec_embedding_mask'].to(model.device)
                    attention_mask = batch['attention_mask'].to(model.device)
                    codec_0_labels = batch['codec_0_labels'].to(model.device)
                    codec_mask = batch['codec_mask'].to(model.device)

                    with accelerator.autocast():
                        unwrap_model = accelerator.unwrap_model(model)
                        speaker_embedding = unwrap_model.speaker_encoder(ref_mels.to(unwrap_model.dtype)).detach()
                        if target_speaker_embedding is None:
                            target_speaker_embedding = speaker_embedding.cpu()

                        input_text_ids = input_ids[:, :, 0]
                        input_codec_ids = input_ids[:, :, 1]

                        input_text_embedding = unwrap_model.talker.text_projection(
                            unwrap_model.talker.get_text_embeddings()(input_text_ids)
                        ) * text_embedding_mask
                        input_codec_embedding = unwrap_model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                        input_codec_embedding[:, 6, :] = speaker_embedding

                        input_embeddings = input_text_embedding + input_codec_embedding

                        for i in range(1, 16):
                            codec_i_embedding = unwrap_model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                            input_embeddings = input_embeddings + codec_i_embedding

                        outputs = unwrap_model.talker(
                            inputs_embeds=input_embeddings[:, :-1, :],
                            attention_mask=attention_mask[:, :-1],
                            labels=codec_0_labels[:, 1:],
                            output_hidden_states=True
                        )

                        hidden_states = outputs.hidden_states[0][-1]
                        talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                        talker_codec_ids = codec_ids[codec_mask]

                        sub_talker_logits, sub_talker_loss = unwrap_model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                        loss = outputs.loss + 0.3 * sub_talker_loss

                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                global_step += 1

                if step % 10 == 0:
                    yield format_train_progress(epoch, step, loss.item())
                    # Log more detailed metrics
                    accelerator.log({
                        "loss": loss.item(),
                        "talker_loss": outputs.loss.item(),
                        "sub_talker_loss": sub_talker_loss.item(),
                        "lr": optimizer.param_groups[0]['lr']
                    }, step=global_step)

                # Log Mel-spectrogram periodically (e.g., every 500 steps)
                if step % 500 == 0:
                    if accelerator.is_main_process:
                        try:
                            # ref_mels is (batch, time, 128)
                            mel_vis = plot_spectrogram_to_numpy(ref_mels[0].detach().cpu().float().numpy())
                            tb_tracker = accelerator.get_tracker("tensorboard")
                            tb_tracker.tracker.add_image("ref_mel", mel_vis, global_step, dataformats='HWC')
                        except Exception as e:
                            accelerator.print(f"Error logging Mel to Tensorboard: {e}")

            # Save accelerator state for resuming
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                yield format_train_progress(epoch, "Saving Checkpoint", 0.0)
                accel_state_dir = os.path.join(args.output_model_path, f"accelerate-epoch-{epoch}")
                os.makedirs(accel_state_dir, exist_ok=True)
                accelerator.save_state(accel_state_dir)

            if accelerator.is_main_process:
                output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
                shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

                input_config_file = os.path.join(MODEL_PATH, "config.json")
                output_config_file = os.path.join(output_dir, "config.json")
                with open(input_config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                config_dict["tts_model_type"] = "custom_voice"
                talker_config = config_dict.get("talker_config", {})
                talker_config["spk_id"] = {
                    args.speaker_name: 3000
                }
                talker_config["spk_is_dialect"] = {
                    args.speaker_name: False
                }
                config_dict["talker_config"] = talker_config

                with open(output_config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

                unwrapped_model = accelerator.unwrap_model(model)
                state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

                drop_prefix = "speaker_encoder"
                keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
                for k in keys_to_drop:
                    del state_dict[k]

                # Ensure target_speaker_embedding is available before using it
                if target_speaker_embedding is None:
                    accelerator.print("Warning: target_speaker_embedding was not set during training steps.")

                weight = state_dict['talker.model.codec_embedding.weight']
                state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
                save_path = os.path.join(output_dir, "model.safetensors")
                save_file(state_dict, save_path)

        accelerator.end_training()
        yield {"type": "progress", "progress": 1.0, "desc": "Training completed."}
        yield {"type": "done", "msg": "Training saved successfully."}
        
    except Exception as e:
        yield {"type": "error", "msg": f"Unhandled exception in training: {str(e)}"}
    finally:
        if 'qwen3tts' in locals():
            del qwen3tts
        if 'model' in locals():
            del model
        if 'accelerator' in locals():
            accelerator.free_memory()
            del accelerator
        if 'optimizer' in locals():
            del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
