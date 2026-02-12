import os
from huggingface_hub import snapshot_download

# Define model ID and local directory
MODEL_ID = "PULSE-ECG/PULSE-7B"
LOCAL_DIR = "/home/csc-project-pulso-2026/pulso-ml-service/models/PULSE-7B"

def download_model():
    print(f"Starting download of {MODEL_ID} to {LOCAL_DIR}...")
    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
            resume_download=True
        )
        print("Download complete successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    download_model()
