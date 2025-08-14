#!/usr/bin/env python3
"""
Download SkyReels-V2 models from Hugging Face
"""

import os
import sys
import argparse
from huggingface_hub import snapshot_download

def download_model(model_id, local_dir=None):
    """
    Download a model from Hugging Face Hub
    
    Args:
        model_id: Hugging Face model ID
        local_dir: Local directory to save the model
    """
    print(f"Downloading {model_id}...")
    
    if local_dir is None:
        # Use default location
        local_dir = os.path.join("checkpoints", model_id.split("/")[-1])
    
    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    
    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ Model downloaded to: {path}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # Show first 10 files
                size = os.path.getsize(os.path.join(root, file))
                size_str = f"{size / 1024**3:.2f}GB" if size > 1024**3 else f"{size / 1024**2:.2f}MB"
                print(f'{subindent}{file} ({size_str})')
            if len(files) > 10:
                print(f'{subindent}... and {len(files) - 10} more files')
        
        return path
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download SkyReels-V2 models")
    parser.add_argument(
        "--model",
        type=str,
        default="Skywork/SkyReels-V2-T2V-14B-540P",
        choices=[
            "Skywork/SkyReels-V2-T2V-14B-540P",
            "Skywork/SkyReels-V2-T2V-14B-720P",
            "Skywork/SkyReels-V2-I2V-1.3B-540P",
            "Skywork/SkyReels-V2-I2V-14B-540P",
            "Skywork/SkyReels-V2-I2V-14B-720P",
        ],
        help="Model to download"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: checkpoints/MODEL_NAME)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SkyReels-V2 Model Downloader")
    print("=" * 60)
    print(f"Model: {args.model}")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        model_name = args.model.split("/")[-1]
        # Map to expected local paths
        if model_name == "SkyReels-V2-T2V-14B-540P":
            output_path = "checkpoints/14B-540P"
        elif model_name == "SkyReels-V2-T2V-14B-720P":
            output_path = "checkpoints/14B-720P"
        else:
            output_path = f"checkpoints/{model_name}"
    
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # Check if model already exists
    if os.path.exists(output_path) and os.listdir(output_path):
        print(f"⚠️  Model already exists at {output_path}")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    # Download the model
    result = download_model(args.model, output_path)
    
    if result:
        print("\n" + "=" * 60)
        print("✅ Download complete!")
        print(f"Model saved to: {result}")
        print("\nYou can now use:")
        print(f"  python convert_to_fp8.py --model_path {result}")
        print(f"  python generate_video_fp8.py --model_path {result}")
    else:
        print("\n❌ Download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()