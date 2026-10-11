"""
Utility script to upload the 1.47 GB gpt2_124M_squad_ft.pt checkpoint
to a Hugging Face Model repository.

Usage:
    python upload_checkpoint_to_hf.py --repo your-username/gpt2-squad-rag
"""

import os
import sys
import argparse
from huggingface_hub import HfApi, create_repo, login

def main():
    parser = argparse.ArgumentParser(description="Upload GPT-2 SQuAD checkpoint to Hugging Face Model Hub")
    parser.add_argument("--repo", required=True, help="HF Repo ID (e.g. 'username/gpt2-squad-rag')")
    parser.add_argument("--file", default="checkpoints/gpt2_124M_squad_ft.pt", help="Path to checkpoint file")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[!] Error: Checkpoint file not found at: {args.file}")
        sys.exit(1)

    print(f"[*] Target repository: {args.repo}")
    print(f"[*] Checkpoint file:    {args.file} ({os.path.getsize(args.file) / (1024*1024):.1f} MB)")

    api = HfApi()

    # Step 1: Ensure repository exists
    try:
        create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
        print(f"[+] Model repository '{args.repo}' is ready.")
    except Exception as e:
        print(f"[!] Warning checking repo: {e}")
        print("    If authentication failed, run 'huggingface-cli login' or pass HF_TOKEN.")

    # Step 2: Upload file
    print(f"[*] Uploading {args.file} to Hugging Face Hub (this may take a few minutes)...")
    try:
        api.upload_file(
            path_or_fileobj=args.file,
            path_in_repo=os.path.basename(args.file),
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"\n[+] SUCCESS! File uploaded to: https://huggingface.co/{args.repo}")
        print(f"    In your Hugging Face Space Settings, add:")
        print(f"    HF_MODEL_REPO = {args.repo}")
    except Exception as e:
        print(f"\n[!] Upload failed: {e}")
        print("    Make sure you are logged into Hugging Face ('huggingface-cli login')")

if __name__ == "__main__":
    main()
