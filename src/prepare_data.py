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
import sys
import gc
import torch

from qwen_tts import Qwen3TTSTokenizer
from utils import resolve_path

BATCH_INFER_NUM = 32

def log_progress(progress, desc):
    print(json.dumps({"type": "progress", "progress": progress, "desc": desc}), flush=True)

def log_done(msg):
    print(json.dumps({"type": "done", "msg": msg}), flush=True)

def log_error(msg):
    print(json.dumps({"type": "error", "msg": msg}), flush=True)

def run_prepare(device, tokenizer_model_path, input_jsonl, output_jsonl):
    try:
        yield {"type": "progress", "progress": 0.05, "desc": f"Loading Tokenizer: {tokenizer_model_path}..."}
        tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_model_path,
            device_map=device,
        )

        total_lines = open(input_jsonl).readlines()
        total_lines = [json.loads(line.strip()) for line in total_lines]
        total_count = len(total_lines)

        final_lines = []
        batch_lines = []
        batch_audios = []
        
        yield {"type": "progress", "progress": 0.1, "desc": f"Starting tokenization of {total_count} files..."}
        
        for idx, line in enumerate(total_lines):
            # Convert to absolute paths for tokenization and robust storage
            line['audio'] = resolve_path(line['audio'])
            if line.get('ref_audio'):
                line['ref_audio'] = resolve_path(line['ref_audio'])
                
            batch_lines.append(line)
            batch_audios.append(line['audio'])

            if len(batch_lines) >= BATCH_INFER_NUM:
                enc_res = tokenizer_12hz.encode(batch_audios)
                for code, item in zip(enc_res.audio_codes, batch_lines):
                    item['audio_codes'] = code.cpu().tolist()
                    final_lines.append(item)
                batch_lines.clear()
                batch_audios.clear()
                
                yield {"type": "progress", "progress": 0.1 + 0.8 * (idx / max(total_count, 1)), "desc": f"Tokenizing: {idx}/{total_count}"}

        if len(batch_audios) > 0:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, item in zip(enc_res.audio_codes, batch_lines):
                item['audio_codes'] = code.cpu().tolist()
                final_lines.append(item)
            batch_lines.clear()
            batch_audios.clear()

        yield {"type": "progress", "progress": 0.95, "desc": "Saving JSONL output..."}
        final_lines = [json.dumps(line, ensure_ascii=False) for line in final_lines]

        with open(output_jsonl, 'w') as f:
            for line in final_lines:
                f.writelines(line + '\n')
                
        yield {"type": "done", "msg": f"Successfully tokenized {len(final_lines)} entries."}
    except Exception as e:
        yield {"type": "error", "msg": f"Error during tokenization: {str(e)}"}
    finally:
        if 'tokenizer_12hz' in locals():
            del tokenizer_12hz
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


