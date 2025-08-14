#!/usr/bin/env python3
"""
Example script showing how to use FP8 quantization with SkyReels V2
"""

import os
import sys

# Example command to run with FP8
print("="*60)
print("SkyReels V2 - FP8 Quantization Example")
print("="*60)
print()

# Check if model path is provided
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = "/home/server/dev/SkyReels-V2-fork/checkpoints/14B-540P"

print(f"Model path: {model_path}")
print()

# Basic generation with FP8
print("1. Basic text-to-video generation with FP8:")
print("-" * 40)
cmd1 = f"""python generate_video.py \\
    --model_id {model_path} \\
    --prompt "A serene lake with mountains" \\
    --use_fp8 \\
    --num_frames 97 \\
    --inference_steps 30"""
print(cmd1)
print()

# FP8 with specific backend
print("2. FP8 with fallback backend (always works):")
print("-" * 40)
cmd2 = f"""python generate_video.py \\
    --model_id {model_path} \\
    --prompt "A beautiful sunset" \\
    --use_fp8 \\
    --fp8_backend fallback \\
    --num_frames 97 \\
    --inference_steps 30"""
print(cmd2)
print()

# Combined optimizations
print("3. FP8 + TeaCache for maximum speed:")
print("-" * 40)
cmd3 = f"""python generate_video.py \\
    --model_id {model_path} \\
    --prompt "Ocean waves at sunset" \\
    --use_fp8 \\
    --teacache \\
    --teacache_thresh 0.2 \\
    --num_frames 97 \\
    --inference_steps 30"""
print(cmd3)
print()

# Test script
print("4. Test FP8 performance:")
print("-" * 40)
cmd4 = f"""python test_fp8_quantization.py \\
    --model_path {model_path} \\
    --fp8_backend auto"""
print(cmd4)
print()

print("="*60)
print("Choose a command above and run it to test FP8!")
print("Note: The warning about transformer_engine is normal.")
print("FP8 will still work using the fallback implementation.")
print("="*60)