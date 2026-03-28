import os
import re

def get_project_root():
    '''Detect the project root directory.
    In Docker, it's usually /workspace.
    Otherwise, it's the parent directory of this src file.
    '''
    if os.path.exists('/.dockerenv') or os.environ.get('IS_DOCKER'):
        return '/workspace'
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_path(path):
    '''Normalize a path to be absolute within the project root if it is relative.'''
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(get_project_root(), path))

def get_model_local_dir(model_id):
    root = get_project_root()
    return os.path.join(root, 'models', model_id)

def is_model_downloaded(model_id):
    local_dir = get_model_local_dir(model_id)
    return os.path.exists(local_dir) and os.path.isdir(local_dir) and len(os.listdir(local_dir)) > 2

def get_model_path(model_id, use_hf=False):
    if os.path.exists(resolve_path(model_id)):
        return resolve_path(model_id)
    output_candidate = resolve_path(os.path.join('output', model_id))
    if os.path.exists(output_candidate):
        print(f'Found local checkpoint at {output_candidate}')
        return output_candidate
    local_dir = get_model_local_dir(model_id)
    if is_model_downloaded(model_id):
        print(f'Found local model at {local_dir}, skipping download!')
        return local_dir
    print(f'Downloading model {model_id} into {local_dir}...')
    os.makedirs(local_dir, exist_ok=True)
    try:
        if use_hf:
            from huggingface_hub import snapshot_download
            return snapshot_download(repo_id=model_id, local_dir=local_dir)
        else:
            from modelscope import snapshot_download
            return snapshot_download(model_id, cache_dir=os.path.join(get_project_root(), 'models'))
    except Exception as e:
        print(f'Warning: Download failed, falling back to id: {e}')
        return model_id

def speaker_key(value):
    return re.sub(r'[^a-z0-9]+', '', str(value).lower())

def resolve_speaker_choice(speaker, supported_speakers):
    if not speaker or not supported_speakers:
        return speaker
    if speaker in supported_speakers:
        return speaker
    lower_map = {str(s).lower(): s for s in supported_speakers}
    lowered = str(speaker).lower()
    if lowered in lower_map:
        return lower_map[lowered]
    normalized = speaker_key(speaker)
    normalized_map = {}
    for s in supported_speakers:
        normalized_map.setdefault(speaker_key(s), s)
    return normalized_map.get(normalized, speaker)
