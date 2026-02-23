import os

def get_project_root():
    """Detect the project root directory. In Docker, it's usually /workspace.
    Otherwise, it's the parent directory of this src file.
    """
    if os.path.exists("/.dockerenv") or os.environ.get("IS_DOCKER"):
        return "/workspace"
    # Assume project root is one level up from this file
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_path(path):
    """Normalize a path to be absolute within the project root if it is relative."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(get_project_root(), path))

def get_model_path(model_id, use_hf=False):
    # Resolve the model_id if it's a relative path candidate
    if os.path.exists(resolve_path(model_id)):
        return resolve_path(model_id)
        
    # Check if it's a checkpoint in the output directory
    output_candidate = resolve_path(os.path.join("output", model_id))
    if os.path.exists(output_candidate):
        print(f"Found local checkpoint at {output_candidate}")
        return output_candidate
        
    # Build explicitly mapped cache directory path
    root = get_project_root()
    local_dir = os.path.join(root, "models", model_id)
    if os.path.exists(local_dir) and os.path.isdir(local_dir) and len(os.listdir(local_dir)) > 2:
        print(f"Found local model at {local_dir}, skipping download!")
        return local_dir
        
    print(f"Downloading model {model_id} into {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        if use_hf:
            from huggingface_hub import snapshot_download
            return snapshot_download(repo_id=model_id, local_dir=local_dir)
        else:
            from modelscope import snapshot_download
            return snapshot_download(model_id, cache_dir=os.path.join(root, "models"))
    except Exception as e:
        print(f"Warning: Download failed, falling back to id: {e}")
        return model_id
