#!/usr/bin/env python3
"""
Test script for FP8 quantization in SkyReels V2
This script validates the FP8 implementation and compares performance metrics.
"""

import argparse
import gc
import os
import time
import torch
import numpy as np
from typing import Dict, Any
import psutil
import GPUtil

from skyreels_v2_infer.modules import (
    get_transformer,
    enable_fp8_quantization,
    get_model_memory_usage,
    download_model
)
from skyreels_v2_infer.pipelines import Text2VideoPipeline


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage"""
    gpus = GPUtil.getGPUs()
    if not gpus:
        return {"used_mb": 0, "free_mb": 0, "total_mb": 0}
    
    gpu = gpus[0]  # Use first GPU
    return {
        "used_mb": gpu.memoryUsed,
        "free_mb": gpu.memoryFree,
        "total_mb": gpu.memoryTotal,
        "utilization": gpu.memoryUtil * 100
    }


def get_system_memory_info() -> Dict[str, float]:
    """Get current system memory usage"""
    mem = psutil.virtual_memory()
    return {
        "used_gb": mem.used / (1024**3),
        "available_gb": mem.available / (1024**3),
        "total_gb": mem.total / (1024**3),
        "percent": mem.percent
    }


def test_model_loading(model_path: str, use_fp8: bool = False, fp8_backend: str = "auto") -> Dict[str, Any]:
    """Test model loading with and without FP8 quantization"""
    print(f"\n{'='*60}")
    print(f"Testing model loading - FP8: {use_fp8}, Backend: {fp8_backend}")
    print(f"{'='*60}")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get initial memory state
    initial_gpu_mem = get_gpu_memory_info()
    initial_sys_mem = get_system_memory_info()
    
    print(f"Initial GPU memory: {initial_gpu_mem['used_mb']:.2f} MB / {initial_gpu_mem['total_mb']:.2f} MB")
    print(f"Initial System memory: {initial_sys_mem['used_gb']:.2f} GB / {initial_sys_mem['total_gb']:.2f} GB")
    
    # Load model
    start_time = time.time()
    transformer = get_transformer(
        model_path,
        device="cuda",
        weight_dtype=torch.bfloat16,
        use_fp8=use_fp8,
        fp8_backend=fp8_backend
    )
    load_time = time.time() - start_time
    
    # Get memory after loading
    loaded_gpu_mem = get_gpu_memory_info()
    loaded_sys_mem = get_system_memory_info()
    
    # Get model memory statistics
    model_stats = get_model_memory_usage(transformer)
    
    # Calculate memory usage
    gpu_mem_used = loaded_gpu_mem['used_mb'] - initial_gpu_mem['used_mb']
    sys_mem_used = loaded_sys_mem['used_gb'] - initial_sys_mem['used_gb']
    
    results = {
        "load_time": load_time,
        "gpu_memory_used_mb": gpu_mem_used,
        "system_memory_used_gb": sys_mem_used,
        "model_stats": model_stats,
        "gpu_memory_after": loaded_gpu_mem,
        "system_memory_after": loaded_sys_mem
    }
    
    print(f"\nModel Loading Results:")
    print(f"  Load time: {load_time:.2f} seconds")
    print(f"  GPU memory used: {gpu_mem_used:.2f} MB")
    print(f"  System memory used: {sys_mem_used:.2f} GB")
    print(f"  Total model parameters: {model_stats['total_params']:,}")
    print(f"  Model memory (theoretical): {model_stats['total_memory_mb']:.2f} MB")
    
    if use_fp8:
        print(f"  FP8 parameters: {model_stats['fp8_params']:,}")
        print(f"  FP8 memory: {model_stats['fp8_memory_mb']:.2f} MB")
    
    # Clean up
    del transformer
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


def test_inference_speed(model_path: str, use_fp8: bool = False, fp8_backend: str = "auto", num_steps: int = 10) -> Dict[str, Any]:
    """Test inference speed with and without FP8 quantization"""
    print(f"\n{'='*60}")
    print(f"Testing inference speed - FP8: {use_fp8}, Steps: {num_steps}")
    print(f"{'='*60}")
    
    # Initialize pipeline
    pipe = Text2VideoPipeline(
        model_path=model_path,
        dit_path=model_path,
        use_fp8=use_fp8,
        fp8_backend=fp8_backend,
        offload=False
    )
    
    # Prepare test input
    test_prompt = "A beautiful sunset over the ocean with waves"
    negative_prompt = "low quality, blurry"
    
    # Warm-up run
    print("Running warm-up inference...")
    with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
        _ = pipe(
            prompt=test_prompt,
            negative_prompt=negative_prompt,
            num_frames=33,  # Smaller for testing
            height=544,
            width=960,
            num_inference_steps=5,  # Few steps for warm-up
            guidance_scale=6.0,
            shift=8.0,
            generator=torch.Generator(device="cuda").manual_seed(42)
        )
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # Benchmark run
    print(f"Running benchmark with {num_steps} inference steps...")
    start_time = time.time()
    
    with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
        torch.cuda.synchronize()
        inference_start = time.time()
        
        _ = pipe(
            prompt=test_prompt,
            negative_prompt=negative_prompt,
            num_frames=33,
            height=544,
            width=960,
            num_inference_steps=num_steps,
            guidance_scale=6.0,
            shift=8.0,
            generator=torch.Generator(device="cuda").manual_seed(42)
        )
        
        torch.cuda.synchronize()
        inference_time = time.time() - inference_start
    
    total_time = time.time() - start_time
    
    # Get memory usage during inference
    inference_gpu_mem = get_gpu_memory_info()
    
    results = {
        "inference_time": inference_time,
        "total_time": total_time,
        "time_per_step": inference_time / num_steps,
        "gpu_memory_during_inference": inference_gpu_mem
    }
    
    print(f"\nInference Results:")
    print(f"  Total inference time: {inference_time:.2f} seconds")
    print(f"  Time per step: {results['time_per_step']:.3f} seconds")
    print(f"  GPU memory during inference: {inference_gpu_mem['used_mb']:.2f} MB")
    print(f"  GPU utilization: {inference_gpu_mem['utilization']:.1f}%")
    
    # Clean up
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


def compare_results(baseline_results: Dict[str, Any], fp8_results: Dict[str, Any]):
    """Compare baseline and FP8 results"""
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    # Memory comparison
    baseline_mem = baseline_results['loading']['gpu_memory_used_mb']
    fp8_mem = fp8_results['loading']['gpu_memory_used_mb']
    mem_reduction = (baseline_mem - fp8_mem) / baseline_mem * 100 if baseline_mem > 0 else 0
    
    print(f"\nMemory Usage:")
    print(f"  Baseline (BF16): {baseline_mem:.2f} MB")
    print(f"  FP8 Quantized: {fp8_mem:.2f} MB")
    print(f"  Memory Reduction: {mem_reduction:.1f}%")
    
    # Model size comparison
    baseline_model_mem = baseline_results['loading']['model_stats']['total_memory_mb']
    fp8_model_mem = fp8_results['loading']['model_stats']['total_memory_mb']
    model_size_reduction = (baseline_model_mem - fp8_model_mem) / baseline_model_mem * 100 if baseline_model_mem > 0 else 0
    
    print(f"\nModel Size (Theoretical):")
    print(f"  Baseline: {baseline_model_mem:.2f} MB")
    print(f"  FP8: {fp8_model_mem:.2f} MB")
    print(f"  Size Reduction: {model_size_reduction:.1f}%")
    
    # Speed comparison
    if 'inference' in baseline_results and 'inference' in fp8_results:
        baseline_time = baseline_results['inference']['time_per_step']
        fp8_time = fp8_results['inference']['time_per_step']
        speedup = baseline_time / fp8_time if fp8_time > 0 else 0
        
        print(f"\nInference Speed:")
        print(f"  Baseline time per step: {baseline_time:.3f} seconds")
        print(f"  FP8 time per step: {fp8_time:.3f} seconds")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Memory during inference
        baseline_inf_mem = baseline_results['inference']['gpu_memory_during_inference']['used_mb']
        fp8_inf_mem = fp8_results['inference']['gpu_memory_during_inference']['used_mb']
        inf_mem_reduction = (baseline_inf_mem - fp8_inf_mem) / baseline_inf_mem * 100 if baseline_inf_mem > 0 else 0
        
        print(f"\nMemory During Inference:")
        print(f"  Baseline: {baseline_inf_mem:.2f} MB")
        print(f"  FP8: {fp8_inf_mem:.2f} MB")
        print(f"  Reduction: {inf_mem_reduction:.1f}%")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Memory reduction: {mem_reduction:.1f}%")
    print(f"✓ Model size reduction: {model_size_reduction:.1f}%")
    if 'inference' in baseline_results and 'inference' in fp8_results:
        print(f"✓ Inference speedup: {speedup:.2f}x")
        print(f"✓ Inference memory reduction: {inf_mem_reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Test FP8 quantization for SkyReels V2")
    parser.add_argument("--model_path", type=str, default="/home/server/dev/SkyReels-V2-fork/checkpoints/14B-540P",
                        help="Path to the model checkpoint")
    parser.add_argument("--test_inference", action="store_true",
                        help="Test inference speed (requires more memory)")
    parser.add_argument("--num_inference_steps", type=int, default=10,
                        help="Number of inference steps for benchmarking")
    parser.add_argument("--fp8_backend", type=str, default="auto",
                        choices=["auto", "native", "transformer_engine", "fallback"],
                        help="FP8 backend to use")
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Downloading model from {args.model_path}...")
        args.model_path = download_model(args.model_path)
    
    print(f"Using model: {args.model_path}")
    print(f"FP8 backend: {args.fp8_backend}")
    
    # Test baseline (BF16)
    print("\n" + "="*60)
    print("TESTING BASELINE (BF16)")
    print("="*60)
    
    baseline_results = {
        'loading': test_model_loading(args.model_path, use_fp8=False)
    }
    
    if args.test_inference:
        baseline_results['inference'] = test_inference_speed(
            args.model_path,
            use_fp8=False,
            num_steps=args.num_inference_steps
        )
    
    # Test with FP8
    print("\n" + "="*60)
    print("TESTING WITH FP8 QUANTIZATION")
    print("="*60)
    
    fp8_results = {
        'loading': test_model_loading(args.model_path, use_fp8=True, fp8_backend=args.fp8_backend)
    }
    
    if args.test_inference:
        fp8_results['inference'] = test_inference_speed(
            args.model_path,
            use_fp8=True,
            fp8_backend=args.fp8_backend,
            num_steps=args.num_inference_steps
        )
    
    # Compare results
    compare_results(baseline_results, fp8_results)
    
    print("\n✅ FP8 quantization test completed successfully!")


if __name__ == "__main__":
    main()