import os

def get_model_path(model_id, use_hf=False):
    # If the user passed a path that exists physically or starts with local output, use directly.
    if model_id.startswith("output/") or model_id.startswith("models/") or model_id.startswith("/workspace/") or os.path.exists(model_id):
        return model_id
        
    # Build explicitly mapped cache directory path
    local_dir = os.path.join("/workspace/models", model_id)
    if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 3: # Basic check to ensure it's not empty
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
            return snapshot_download(model_id, local_dir=local_dir)
    except Exception as e:
        print(f"Warning: Download failed or syntax deprecated, falling back to default hub download: {e}")
        return model_id
