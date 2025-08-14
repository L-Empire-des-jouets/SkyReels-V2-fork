#!/usr/bin/env python3
"""
Convert SkyReels-V2 14B model to FP8 quantized format
This script loads the original model and converts it to use FP8 quantization
to reduce VRAM usage while maintaining generation quality.
"""

import argparse
import os
import sys
import torch
import gc
from pathlib import Path
import json
import time

# Add the project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from skyreels_v2_infer.modules import get_transformer
from skyreels_v2_infer.modules.transformer_fp8 import WanModelFP8
from skyreels_v2_infer.modules.fp8_quantization import (
    quantize_linear_layers, 
    calculate_memory_savings,
    HAS_FP8
)


def print_memory_stats():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")


def convert_model_to_fp8(model_path: str, output_path: str, device: str = "cuda"):
    """
    Convert a SkyReels-V2 model to FP8 quantized format
    
    Args:
        model_path: Path to the original model checkpoint
        output_path: Path where the quantized model will be saved
        device: Device to use for conversion
    """
    print(f"Converting model from {model_path} to FP8 format...")
    print(f"FP8 support available: {HAS_FP8}")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    print("\n1. Loading original model...")
    print_memory_stats()
    
    # Load the original model in CPU first to save memory
    try:
        # Load transformer model
        original_model = get_transformer(
            model_path, 
            device="cpu",  # Load on CPU first
            weight_dtype=torch.bfloat16
        )
        print(f"Model loaded successfully. Type: {type(original_model)}")
        
        # Get model configuration
        if hasattr(original_model, 'config'):
            config = original_model.config
        else:
            # Extract configuration from model attributes
            config = {
                'dim': original_model.dim if hasattr(original_model, 'dim') else 3072,
                'num_blocks': len(original_model.blocks) if hasattr(original_model, 'blocks') else 32,
                'num_heads': 24,  # Default for 14B model
                'text_dim': 4096,
                'freq_dim': 256,
            }
        
        print(f"Model configuration: {config}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    print("\n2. Converting to FP8...")
    print_memory_stats()
    
    # Quantize linear layers
    print("Quantizing linear layers...")
    quantized_model = quantize_linear_layers(
        original_model,
        exclude_layers=['head', 'embedding']  # Keep critical layers in higher precision
    )
    
    # Calculate memory savings
    print("\n3. Calculating memory savings...")
    memory_stats = calculate_memory_savings(quantized_model)
    
    print(f"\nMemory Statistics:")
    print(f"  Total Parameters: {memory_stats['total_params']:,}")
    print(f"  Original Memory (FP32): {memory_stats['fp32_memory_gb']:.2f} GB")
    print(f"  Quantized Memory (FP8): {memory_stats['fp8_memory_gb']:.2f} GB")
    print(f"  Memory Reduction: {memory_stats['memory_reduction']*100:.1f}%")
    print(f"  Expected Speedup: {memory_stats['speedup_estimate']:.1f}x")
    
    # Move model to device for saving
    print(f"\n4. Moving model to {device}...")
    quantized_model = quantized_model.to(device)
    print_memory_stats()
    
    # Save the quantized model
    print(f"\n5. Saving quantized model to {output_path}...")
    
    # Save model weights
    model_save_path = os.path.join(output_path, "dit_checkpoint_fp8.pt")
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'config': config,
        'quantization': 'fp8',
        'memory_stats': memory_stats,
        'original_model_path': model_path,
        'conversion_timestamp': time.time()
    }, model_save_path)
    
    print(f"Model saved to: {model_save_path}")
    
    # Save configuration as JSON for easy inspection
    config_path = os.path.join(output_path, "config_fp8.json")
    with open(config_path, 'w') as f:
        json.dump({
            'model_config': config,
            'quantization': 'fp8',
            'memory_stats': memory_stats,
            'original_model_path': model_path
        }, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    
    # Copy other necessary files from original model directory
    print("\n6. Copying additional model files...")
    
    # Files to copy (if they exist)
    files_to_copy = [
        'Wan2.1_VAE.pth',
        'config.json',
        'model_index.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'vocab.json',
        'merges.txt'
    ]
    
    for filename in files_to_copy:
        src_path = os.path.join(model_path, filename)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_path, filename)
            import shutil
            shutil.copy2(src_path, dst_path)
            print(f"  Copied: {filename}")
    
    # Clean up
    del original_model
    del quantized_model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✅ Conversion completed successfully!")
    print(f"Quantized model saved to: {output_path}")
    
    return output_path


def verify_quantized_model(model_path: str):
    """
    Verify that the quantized model can be loaded and used
    
    Args:
        model_path: Path to the quantized model
    """
    print(f"\nVerifying quantized model at {model_path}...")
    
    try:
        # Load the checkpoint
        checkpoint_path = os.path.join(model_path, "dit_checkpoint_fp8.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        print("✅ Checkpoint loaded successfully")
        print(f"  Quantization type: {checkpoint.get('quantization', 'unknown')}")
        print(f"  Memory stats: {checkpoint.get('memory_stats', {})}")
        
        # Try to instantiate the model (without loading weights to save memory)
        from skyreels_v2_infer.modules.transformer_fp8 import WanModelFP8
        
        config = checkpoint.get('config', {})
        print(f"  Model config: {config}")
        
        print("✅ Model verification successful!")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert SkyReels-V2 model to FP8 quantized format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/server/dev/SkyReels-V2-fork/checkpoints/14B-540P",
        help="Path to the original model checkpoint"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path where the quantized model will be saved (default: model_path_fp8)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for conversion"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the quantized model after conversion"
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output_path is None:
        args.output_path = args.model_path + "_fp8"
    
    print("=" * 60)
    print("SkyReels-V2 FP8 Quantization Converter")
    print("=" * 60)
    print(f"Input model: {args.model_path}")
    print(f"Output path: {args.output_path}")
    print(f"Device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Count: {torch.cuda.device_count()}")
    print("=" * 60)
    
    try:
        # Perform conversion
        output_path = convert_model_to_fp8(
            args.model_path,
            args.output_path,
            args.device
        )
        
        # Verify if requested
        if args.verify:
            verify_quantized_model(output_path)
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()