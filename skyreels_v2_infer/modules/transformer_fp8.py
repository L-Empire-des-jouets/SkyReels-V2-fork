"""
FP8 Quantized Transformer for SkyReels-V2
Implements the DiT transformer with FP8 quantization and SageAttention
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .transformer import WanModel, WanRMSNorm, WanBlock, WanSelfAttention
from .fp8_quantization import FP8LinearQuantized, SageAttention8Bit, quantize_linear_layers


class WanSelfAttentionFP8(nn.Module):
    """FP8 Quantized Self-Attention module"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, rope_applied=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope_applied = rope_applied
        
        # Use FP8 quantized linear layers
        self.q = FP8LinearQuantized(dim, dim, bias=qkv_bias)
        self.k = FP8LinearQuantized(dim, dim, bias=qkv_bias)
        self.v = FP8LinearQuantized(dim, dim, bias=qkv_bias)
        self.o = FP8LinearQuantized(dim, dim, bias=True)
        
        # For 8-bit attention computation
        self.use_sage_attention = True
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, grid_sizes=None, freqs=None):
        B, L, C = x.shape
        
        # Compute Q, K, V with FP8 layers
        q = self.q(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k(x).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v(x).reshape(B, L, self.num_heads, self.head_dim)
        
        # Apply RoPE if needed (keeping original logic)
        if self.rope_applied and freqs is not None:
            from .transformer import rope_apply
            q = rope_apply(q.transpose(1, 2), grid_sizes, freqs).transpose(1, 2)
            k = rope_apply(k.transpose(1, 2), grid_sizes, freqs).transpose(1, 2)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # (B, num_heads, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention with 8-bit quantization for memory efficiency
        if self.use_sage_attention and not self.training:
            # Quantize for attention computation
            attn_output = self._sage_attention(q, k, v)
        else:
            # Standard attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, L, C)
        output = self.o(attn_output)
        
        return output
    
    def _sage_attention(self, q, k, v):
        """Apply SageAttention 8-bit quantization"""
        # Quantize tensors to INT8
        def quantize_int8(tensor):
            scale = tensor.abs().max() / 127.0
            quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
            return quantized, scale
        
        q_int8, q_scale = quantize_int8(q)
        k_int8, k_scale = quantize_int8(k)
        
        # Compute attention scores (dequantize for matmul)
        q_dequant = q_int8.float() * q_scale
        k_dequant = k_int8.float() * k_scale
        attn_scores = torch.matmul(q_dequant, k_dequant.transpose(-2, -1)) * self.scale
        
        # Softmax in FP32 for stability
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(v.dtype)
        
        # Quantize attention weights and values
        attn_int8, attn_scale = quantize_int8(attn_probs)
        v_int8, v_scale = quantize_int8(v)
        
        # Compute output
        attn_dequant = attn_int8.float() * attn_scale
        v_dequant = v_int8.float() * v_scale
        attn_output = torch.matmul(attn_dequant, v_dequant)
        
        return attn_output


class WanBlockFP8(nn.Module):
    """FP8 Quantized Transformer Block"""
    
    def __init__(self, dim, num_heads=8, ffn_dim_multiplier=4, qkv_bias=False, rope_applied=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Layer normalization (keep in FP32 for stability)
        self.norm1 = WanRMSNorm(dim)
        self.norm2 = WanRMSNorm(dim)
        
        # FP8 Quantized Self-Attention
        self.self_attn = WanSelfAttentionFP8(dim, num_heads, qkv_bias, rope_applied)
        
        # FP8 Quantized FFN
        ffn_dim = int(dim * ffn_dim_multiplier)
        self.ffn = nn.Sequential(
            FP8LinearQuantized(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            FP8LinearQuantized(ffn_dim, dim)
        )
        
    def forward(self, x, grid_sizes=None, freqs=None):
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, grid_sizes, freqs)
        x = residual + x
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class WanModelFP8(WanModel):
    """
    FP8 Quantized version of WanModel
    Inherits from original WanModel and overrides specific components
    """
    
    def __init__(self, *args, use_fp8=True, use_sage_attention=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_fp8 = use_fp8
        self.use_sage_attention = use_sage_attention
        
        if use_fp8:
            self._convert_to_fp8()
    
    def _convert_to_fp8(self):
        """Convert the model to use FP8 quantization"""
        
        # Replace transformer blocks with FP8 versions
        new_blocks = nn.ModuleList()
        for block in self.blocks:
            fp8_block = WanBlockFP8(
                dim=block.dim,
                num_heads=block.self_attn.num_heads if hasattr(block.self_attn, 'num_heads') else 8,
                ffn_dim_multiplier=4,
                qkv_bias=False,
                rope_applied=block.self_attn.rope_applied if hasattr(block.self_attn, 'rope_applied') else False
            )
            
            # Copy weights from original block
            self._copy_block_weights(block, fp8_block)
            new_blocks.append(fp8_block)
        
        self.blocks = new_blocks
        
        # Quantize other linear layers
        self._quantize_embeddings()
        self._quantize_head()
    
    def _copy_block_weights(self, src_block, dst_block):
        """Copy weights from source block to FP8 destination block"""
        # Copy attention weights
        if hasattr(src_block.self_attn, 'q') and hasattr(src_block.self_attn.q, 'weight'):
            dst_block.self_attn.q.quantize_weight(src_block.self_attn.q.weight.data)
            if src_block.self_attn.q.bias is not None:
                dst_block.self_attn.q.bias.data = src_block.self_attn.q.bias.data.clone()
        
        if hasattr(src_block.self_attn, 'k') and hasattr(src_block.self_attn.k, 'weight'):
            dst_block.self_attn.k.quantize_weight(src_block.self_attn.k.weight.data)
            if src_block.self_attn.k.bias is not None:
                dst_block.self_attn.k.bias.data = src_block.self_attn.k.bias.data.clone()
        
        if hasattr(src_block.self_attn, 'v') and hasattr(src_block.self_attn.v, 'weight'):
            dst_block.self_attn.v.quantize_weight(src_block.self_attn.v.weight.data)
            if src_block.self_attn.v.bias is not None:
                dst_block.self_attn.v.bias.data = src_block.self_attn.v.bias.data.clone()
        
        if hasattr(src_block.self_attn, 'o') and hasattr(src_block.self_attn.o, 'weight'):
            dst_block.self_attn.o.quantize_weight(src_block.self_attn.o.weight.data)
            if src_block.self_attn.o.bias is not None:
                dst_block.self_attn.o.bias.data = src_block.self_attn.o.bias.data.clone()
        
        # Copy FFN weights
        if hasattr(src_block, 'ffn'):
            for i, layer in enumerate(src_block.ffn):
                if isinstance(layer, nn.Linear) and i < len(dst_block.ffn):
                    dst_layer = dst_block.ffn[i]
                    if isinstance(dst_layer, FP8LinearQuantized):
                        dst_layer.quantize_weight(layer.weight.data)
                        if layer.bias is not None:
                            dst_layer.bias.data = layer.bias.data.clone()
        
        # Copy normalization weights
        dst_block.norm1.weight.data = src_block.norm1.weight.data.clone()
        dst_block.norm2.weight.data = src_block.norm2.weight.data.clone()
    
    def _quantize_embeddings(self):
        """Quantize embedding layers to FP8"""
        # Text embedding
        if hasattr(self, 'text_embedding'):
            for i, layer in enumerate(self.text_embedding):
                if isinstance(layer, nn.Linear):
                    fp8_layer = FP8LinearQuantized.from_linear(layer)
                    self.text_embedding[i] = fp8_layer
        
        # Time embedding
        if hasattr(self, 'time_embedding'):
            for i, layer in enumerate(self.time_embedding):
                if isinstance(layer, nn.Linear):
                    fp8_layer = FP8LinearQuantized.from_linear(layer)
                    self.time_embedding[i] = fp8_layer
        
        # Time projection
        if hasattr(self, 'time_projection'):
            for i, layer in enumerate(self.time_projection):
                if isinstance(layer, nn.Linear):
                    fp8_layer = FP8LinearQuantized.from_linear(layer)
                    self.time_projection[i] = fp8_layer
    
    def _quantize_head(self):
        """Quantize the output head to FP8"""
        if hasattr(self, 'head') and isinstance(self.head, nn.Linear):
            self.head = FP8LinearQuantized.from_linear(self.head)
    
    @classmethod
    def from_pretrained(cls, model_path: str, use_fp8: bool = True, use_sage_attention: bool = True, **kwargs):
        """
        Load a pretrained model and optionally convert to FP8
        
        Args:
            model_path: Path to the pretrained model
            use_fp8: Whether to use FP8 quantization
            use_sage_attention: Whether to use SageAttention
        """
        # Load the original model
        model = super().from_pretrained(model_path, **kwargs)
        
        # Convert to FP8 if requested
        if use_fp8:
            fp8_model = cls(
                model.config if hasattr(model, 'config') else {},
                use_fp8=True,
                use_sage_attention=use_sage_attention
            )
            
            # Copy state dict
            fp8_model.load_state_dict(model.state_dict(), strict=False)
            
            return fp8_model
        
        return model
    
    def estimate_memory_usage(self):
        """Estimate memory usage of the quantized model"""
        total_params = sum(p.numel() for p in self.parameters())
        
        # Count FP8 vs regular parameters
        fp8_params = 0
        regular_params = 0
        
        for name, module in self.named_modules():
            if isinstance(module, FP8LinearQuantized):
                fp8_params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, nn.Linear):
                regular_params += sum(p.numel() for p in module.parameters())
        
        # Calculate memory (FP8 = 1 byte, BF16 = 2 bytes, FP32 = 4 bytes)
        fp8_memory_gb = fp8_params / (1024**3)
        regular_memory_gb = regular_params * 2 / (1024**3)  # Assuming BF16
        total_memory_gb = fp8_memory_gb + regular_memory_gb
        
        return {
            'total_params': total_params,
            'fp8_params': fp8_params,
            'regular_params': regular_params,
            'total_memory_gb': total_memory_gb,
            'memory_reduction': 1 - (total_memory_gb / (total_params * 2 / (1024**3)))
        }


# Export the FP8 model
__all__ = ['WanModelFP8', 'WanBlockFP8', 'WanSelfAttentionFP8']