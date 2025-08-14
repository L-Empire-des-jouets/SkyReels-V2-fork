#!/usr/bin/env python3
"""
Generate videos using FP8 quantized SkyReels-V2 models
This script uses FP8 quantization to reduce VRAM usage and enable
video generation on GPUs with limited memory.
"""

import argparse
import gc
import os
import random
import time
import sys

import imageio
import torch
from diffusers.utils import load_image

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines.text2video_pipeline_fp8 import Text2VideoPipelineFP8
from skyreels_v2_infer.pipelines import PromptEnhancer
from skyreels_v2_infer.pipelines import resizecrop


def print_memory_stats(label=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{label} GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos using FP8 quantized SkyReels-V2 models"
    )
    
    # Model arguments
    parser.add_argument("--model_path", type=str, 
                       default="/home/server/dev/SkyReels-V2-fork/checkpoints/14B-540P",
                       help="Path to the model checkpoint")
    parser.add_argument("--use_fp8", action="store_true", default=True,
                       help="Use FP8 quantization (default: True)")
    parser.add_argument("--use_sage_attention", action="store_true", default=True,
                       help="Use SageAttention for additional memory savings")
    
    # Generation arguments
    parser.add_argument("--prompt", type=str,
                       default="A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface.",
                       help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str,
                       default="Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality",
                       help="Negative prompt")
    parser.add_argument("--outdir", type=str, default="video_out_fp8",
                       help="Output directory for generated videos")
    
    # Video parameters
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"], default="540P",
                       help="Video resolution")
    parser.add_argument("--num_frames", type=int, default=97,
                       help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=24,
                       help="Frames per second for output video")
    
    # Generation parameters
    parser.add_argument("--guidance_scale", type=float, default=6.0,
                       help="Guidance scale for CFG")
    parser.add_argument("--shift", type=float, default=8.0,
                       help="Shift parameter for scheduler")
    parser.add_argument("--inference_steps", type=int, default=30,
                       help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    # Optimization arguments
    parser.add_argument("--offload", action="store_true",
                       help="Offload models to CPU when not in use")
    parser.add_argument("--use_usp", action="store_true",
                       help="Use Unified Sequence Parallel (requires multi-GPU)")
    parser.add_argument("--prompt_enhancer", action="store_true",
                       help="Use prompt enhancer")
    parser.add_argument("--memory_efficient_attention", action="store_true", default=True,
                       help="Use memory efficient attention")
    
    # Monitoring
    parser.add_argument("--monitor_memory", action="store_true",
                       help="Monitor memory usage during generation")
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is None:
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))
    
    print("=" * 60)
    print("SkyReels-V2 FP8 Video Generation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"FP8 Quantization: {args.use_fp8}")
    print(f"SageAttention: {args.use_sage_attention}")
    print(f"Resolution: {args.resolution}")
    print(f"Frames: {args.num_frames}")
    print(f"Steps: {args.inference_steps}")
    print(f"Seed: {args.seed}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Count: {torch.cuda.device_count()}")
    print("=" * 60)
    
    # Set resolution
    if args.resolution == "540P":
        height, width = 544, 960
    elif args.resolution == "720P":
        height, width = 720, 1280
    else:
        raise ValueError(f"Invalid resolution: {args.resolution}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Handle USP setup if requested
    local_rank = 0
    if args.use_usp:
        assert not args.prompt_enhancer, "Prompt enhancer not compatible with USP"
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        import torch.distributed as dist
        
        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(dist.get_rank())
        device = "cuda"
        
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
    
    # Enhance prompt if requested
    prompt_input = args.prompt
    if args.prompt_enhancer:
        print("\nEnhancing prompt...")
        prompt_enhancer = PromptEnhancer()
        prompt_input = prompt_enhancer(prompt_input)
        print(f"Enhanced prompt: {prompt_input}")
        del prompt_enhancer
        gc.collect()
        torch.cuda.empty_cache()
    
    # Initialize FP8 pipeline
    print("\nInitializing FP8 pipeline...")
    print_memory_stats("Before pipeline init:")
    
    # Check if we have an FP8 checkpoint
    fp8_model_path = args.model_path + "_fp8"
    if os.path.exists(fp8_model_path):
        print(f"Using pre-converted FP8 model at {fp8_model_path}")
        model_path = fp8_model_path
    else:
        model_path = args.model_path
    
    pipe = Text2VideoPipelineFP8(
        model_path=model_path,
        dit_path=model_path,
        device=device,
        weight_dtype=torch.bfloat16,
        use_usp=args.use_usp,
        offload=args.offload,
        use_fp8=args.use_fp8,
        use_sage_attention=args.use_sage_attention
    )
    
    print_memory_stats("After pipeline init:")
    
    # Prepare generation kwargs
    kwargs = {
        "prompt": prompt_input,
        "negative_prompt": args.negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "generator": torch.Generator(device=device).manual_seed(args.seed),
        "height": height,
        "width": width,
        "enable_memory_efficient_attention": args.memory_efficient_attention,
    }
    
    # Create output directory
    save_dir = os.path.join("result", args.outdir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate video
    print(f"\nGenerating video...")
    print(f"Parameters: {kwargs}")
    
    if args.monitor_memory:
        print("\nMemory monitoring enabled - will print stats during generation")
    
    start_time = time.time()
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
        if args.monitor_memory:
            # Monitor memory during generation
            import threading
            stop_monitoring = threading.Event()
            
            def monitor():
                step = 0
                while not stop_monitoring.is_set():
                    print_memory_stats(f"Step {step}:")
                    step += 1
                    time.sleep(2)
            
            monitor_thread = threading.Thread(target=monitor)
            monitor_thread.start()
        
        try:
            video_frames = pipe(**kwargs)[0]
        finally:
            if args.monitor_memory:
                stop_monitoring.set()
                monitor_thread.join()
    
    generation_time = time.time() - start_time
    print(f"\nGeneration completed in {generation_time:.2f} seconds")
    print_memory_stats("After generation:")
    
    # Save video (only on rank 0 for multi-GPU)
    if local_rank == 0:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        video_filename = f"fp8_{args.resolution}_{args.seed}_{current_time}.mp4"
        output_path = os.path.join(save_dir, video_filename)
        
        print(f"\nSaving video to {output_path}...")
        imageio.mimwrite(
            output_path, 
            video_frames, 
            fps=args.fps, 
            quality=8, 
            output_params=["-loglevel", "error"]
        )
        print(f"✅ Video saved successfully!")
        
        # Print final statistics
        print("\n" + "=" * 60)
        print("Generation Statistics:")
        print(f"  Output: {output_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Frames: {args.num_frames}")
        print(f"  FPS: {args.fps}")
        print(f"  Generation Time: {generation_time:.2f}s")
        print(f"  Time per frame: {generation_time/args.num_frames:.3f}s")
        print(f"  FP8 Enabled: {args.use_fp8}")
        print(f"  SageAttention: {args.use_sage_attention}")
        print_memory_stats("  Final")
        print("=" * 60)
    
    # Clean up
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()