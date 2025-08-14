#!/usr/bin/env python3
"""
Debug script to understand the model structure
"""

import sys
import os
sys.path.append('/workspace')

from skyreels_v2_infer.modules import get_transformer
import torch

model_path = "/home/server/dev/SkyReels-V2-fork/checkpoints/14B-540P"

print("Loading model...")
model = get_transformer(model_path, device="cpu", weight_dtype=torch.bfloat16)

print(f"\nModel type: {type(model)}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# List all modules
print("\n=== All Modules ===")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"Linear: {name} - shape: {module.weight.shape}")

# Check blocks structure
if hasattr(model, 'blocks'):
    print(f"\n=== Blocks: {len(model.blocks)} ===")
    for i, block in enumerate(model.blocks[:2]):  # Just first 2 blocks
        print(f"\nBlock {i}:")
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(f"  Linear: {name} - shape: {module.weight.shape}")

# Check specific attributes
print("\n=== Model Attributes ===")
for attr in ['head', 'text_embedding', 'time_embedding', 'time_projection', 'fps_projection']:
    if hasattr(model, attr):
        module = getattr(model, attr)
        print(f"{attr}: {type(module)}")
        if isinstance(module, torch.nn.Sequential):
            for i, layer in enumerate(module):
                if isinstance(layer, torch.nn.Linear):
                    print(f"  [{i}] Linear: {layer.weight.shape}")