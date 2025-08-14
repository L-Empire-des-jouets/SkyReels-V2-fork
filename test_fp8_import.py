#!/usr/bin/env python3
"""
Test script to verify FP8 implementation can be imported
"""

import sys
import os

# Add project to path
sys.path.insert(0, '/workspace')

print("Testing FP8 implementation imports...")

try:
    print("1. Testing fp8_quantization module...")
    from skyreels_v2_infer.modules.fp8_quantization import (
        FP8LinearQuantized, 
        SageAttention8Bit,
        quantize_linear_layers,
        HAS_FP8
    )
    print("   ✓ fp8_quantization module imported successfully")
    print(f"   FP8 support available: {HAS_FP8}")
    
except ImportError as e:
    print(f"   ✗ Error importing fp8_quantization: {e}")
    sys.exit(1)

try:
    print("\n2. Testing transformer_fp8 module...")
    from skyreels_v2_infer.modules.transformer_fp8 import (
        WanModelFP8,
        WanAttentionBlockFP8,
        WanSelfAttentionFP8
    )
    print("   ✓ transformer_fp8 module imported successfully")
    
except ImportError as e:
    print(f"   ✗ Error importing transformer_fp8: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n3. Testing text2video_pipeline_fp8 module...")
    from skyreels_v2_infer.pipelines.text2video_pipeline_fp8 import Text2VideoPipelineFP8
    print("   ✓ text2video_pipeline_fp8 module imported successfully")
    
except ImportError as e:
    print(f"   ✗ Error importing text2video_pipeline_fp8: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All FP8 modules imported successfully!")
print("\nYou can now use:")
print("  - python3 convert_to_fp8.py")
print("  - python3 generate_video_fp8.py")