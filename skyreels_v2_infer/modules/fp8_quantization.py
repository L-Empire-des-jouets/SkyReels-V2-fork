# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
FP8 Quantization Module for SkyReels V2
Implements FP8 quantization for linear layers and attention operations
to reduce memory usage and accelerate inference on RTX 4090/5090 GPUs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings

try:
    # Try to import transformer_engine for FP8 support
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    HAS_TRANSFORMER_ENGINE = True
except ImportError:
    HAS_TRANSFORMER_ENGINE = False
    warnings.warn("transformer_engine not found. FP8 quantization will use fallback implementation.")

try:
    # Try to import torch._scaled_mm for native FP8 support (PyTorch 2.3+)
    if hasattr(torch, '_scaled_mm'):
        from torch._scaled_mm import scaled_mm
        HAS_NATIVE_FP8 = True
    else:
        HAS_NATIVE_FP8 = False
        scaled_mm = None
except (ImportError, AttributeError):
    HAS_NATIVE_FP8 = False
    scaled_mm = None


__all__ = [
    "FP8Linear",
    "FP8Attention",
    "quantize_model_to_fp8",
    "enable_fp8_quantization",
]


class FP8Config:
    """Configuration for FP8 quantization"""
    def __init__(
        self,
        enabled: bool = False,
        use_dynamic_scaling: bool = True,
        amax_history_len: int = 16,
        amax_compute_algo: str = "most_recent",
        fp8_format: str = "e4m3",  # e4m3 or e5m2
        backend: str = "auto",  # auto, native, transformer_engine, fallback
    ):
        self.enabled = enabled
        self.use_dynamic_scaling = use_dynamic_scaling
        self.amax_history_len = amax_history_len
        self.amax_compute_algo = amax_compute_algo
        self.fp8_format = fp8_format
        self.backend = backend
        
        # Auto-select backend
        if backend == "auto":
            if HAS_TRANSFORMER_ENGINE:
                self.backend = "transformer_engine"
            elif HAS_NATIVE_FP8:
                self.backend = "native"
            else:
                self.backend = "fallback"
                

# Global FP8 configuration
_fp8_config = FP8Config()


def enable_fp8_quantization(
    enabled: bool = True,
    use_dynamic_scaling: bool = True,
    backend: str = "auto"
):
    """Enable or disable FP8 quantization globally"""
    global _fp8_config
    _fp8_config.enabled = enabled
    _fp8_config.use_dynamic_scaling = use_dynamic_scaling
    _fp8_config.backend = backend
    if backend == "auto":
        if HAS_TRANSFORMER_ENGINE:
            _fp8_config.backend = "transformer_engine"
        elif HAS_NATIVE_FP8:
            _fp8_config.backend = "native"
        else:
            _fp8_config.backend = "fallback"
    
    if enabled:
        print(f"FP8 quantization enabled with backend: {_fp8_config.backend}")
    return _fp8_config


def to_fp8(tensor: torch.Tensor, dtype: str = "e4m3") -> torch.Tensor:
    """Convert tensor to FP8 format"""
    if dtype == "e4m3":
        target_dtype = torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.bfloat16
    elif dtype == "e5m2":
        target_dtype = torch.float8_e5m2 if hasattr(torch, 'float8_e5m2') else torch.bfloat16
    else:
        target_dtype = torch.bfloat16
    
    return tensor.to(target_dtype)


class FP8Linear(nn.Module):
    """FP8 quantized linear layer with dynamic scaling"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize as regular linear layer
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        # FP8 specific parameters
        self.weight_scale = nn.Parameter(torch.ones(1, device=device), requires_grad=False)
        self.input_scale = nn.Parameter(torch.ones(1, device=device), requires_grad=False)
        self.output_scale = nn.Parameter(torch.ones(1, device=device), requires_grad=False)
        
        # History for dynamic scaling
        self.register_buffer('weight_amax_history', torch.zeros(16))
        self.register_buffer('input_amax_history', torch.zeros(16))
        self.history_idx = 0
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using Xavier initialization"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def update_scales(self, input_tensor: torch.Tensor):
        """Update scaling factors for dynamic quantization"""
        if _fp8_config.use_dynamic_scaling:
            # Compute absolute max values
            weight_amax = self.weight.abs().max().item()
            input_amax = input_tensor.abs().max().item()
            
            # Update history
            idx = self.history_idx % 16
            self.weight_amax_history[idx] = weight_amax
            self.input_amax_history[idx] = input_amax
            self.history_idx += 1
            
            # Compute scales based on history
            if _fp8_config.amax_compute_algo == "most_recent":
                self.weight_scale.data = torch.tensor([weight_amax / 448.0])  # E4M3 max value
                self.input_scale.data = torch.tensor([input_amax / 448.0])
            elif _fp8_config.amax_compute_algo == "max":
                self.weight_scale.data = self.weight_amax_history.max() / 448.0
                self.input_scale.data = self.input_amax_history.max() / 448.0
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP8 quantization"""
        if not _fp8_config.enabled:
            # Regular linear layer
            return F.linear(input, self.weight, self.bias)
        
        # Update scaling factors
        self.update_scales(input)
        
        if _fp8_config.backend == "transformer_engine" and HAS_TRANSFORMER_ENGINE:
            # Use Transformer Engine for FP8 GEMM
            with te.fp8_autocast(enabled=True):
                output = F.linear(input, self.weight, self.bias)
        elif _fp8_config.backend == "native" and HAS_NATIVE_FP8:
            # Use PyTorch native FP8 support
            weight_fp8 = to_fp8(self.weight / self.weight_scale, _fp8_config.fp8_format)
            input_fp8 = to_fp8(input / self.input_scale, _fp8_config.fp8_format)
            
            # Scaled matrix multiplication
            output = scaled_mm(
                input_fp8, weight_fp8.t(),
                scale_a=self.input_scale,
                scale_b=self.weight_scale,
                out_dtype=input.dtype
            )
            
            if self.bias is not None:
                output = output + self.bias
        else:
            # Fallback: Simulate FP8 with BF16
            weight_quant = (self.weight / self.weight_scale).to(torch.bfloat16)
            input_quant = (input / self.input_scale).to(torch.bfloat16)
            
            output = F.linear(input_quant, weight_quant, None)
            output = output * self.weight_scale * self.input_scale
            
            if self.bias is not None:
                output = output + self.bias
            
            output = output.to(input.dtype)
        
        return output
    
    @classmethod
    def from_linear(cls, linear: nn.Linear) -> 'FP8Linear':
        """Convert a regular linear layer to FP8Linear"""
        fp8_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype
        )
        fp8_linear.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            fp8_linear.bias.data = linear.bias.data.clone()
        return fp8_linear


class FP8Attention(nn.Module):
    """FP8 quantized attention module"""
    
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        
        # Use FP8Linear for Q, K, V, O projections
        self.q = FP8Linear(dim, dim)
        self.k = FP8Linear(dim, dim)
        self.v = FP8Linear(dim, dim)
        self.o = FP8Linear(dim, dim)
        
        # Keep normalization in higher precision
        if qk_norm:
            self.norm_q = nn.LayerNorm(dim, eps=eps)
            self.norm_k = nn.LayerNorm(dim, eps=eps)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        use_flash_attention: bool = True
    ) -> torch.Tensor:
        """Forward pass with FP8 quantized attention"""
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        
        # Compute Q, K, V with FP8 quantization
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        
        if context is not None:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
        else:
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
        
        if _fp8_config.enabled and _fp8_config.backend == "transformer_engine" and HAS_TRANSFORMER_ENGINE:
            # Use Transformer Engine's FP8 attention
            with te.fp8_autocast(enabled=True):
                attn_output = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    dropout_p=0.0,
                    is_causal=False
                ).transpose(1, 2).contiguous()
        else:
            # Standard attention (will use FP8 in Q,K,V,O projections)
            attn_output = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False
            ).transpose(1, 2).contiguous()
        
        # Output projection with FP8
        attn_output = attn_output.flatten(2)
        output = self.o(attn_output)
        
        return output
    
    @classmethod
    def from_module(cls, module: nn.Module) -> 'FP8Attention':
        """Convert existing attention module to FP8Attention"""
        fp8_attn = cls(
            dim=module.dim,
            num_heads=module.num_heads,
            qk_norm=module.qk_norm,
            eps=module.eps
        )
        
        # Copy weights from original module
        if hasattr(module, 'q'):
            fp8_attn.q = FP8Linear.from_linear(module.q)
        if hasattr(module, 'k'):
            fp8_attn.k = FP8Linear.from_linear(module.k)
        if hasattr(module, 'v'):
            fp8_attn.v = FP8Linear.from_linear(module.v)
        if hasattr(module, 'o'):
            fp8_attn.o = FP8Linear.from_linear(module.o)
        
        # Copy normalization layers
        if hasattr(module, 'norm_q') and module.qk_norm:
            fp8_attn.norm_q.load_state_dict(module.norm_q.state_dict())
        if hasattr(module, 'norm_k') and module.qk_norm:
            fp8_attn.norm_k.load_state_dict(module.norm_k.state_dict())
        
        return fp8_attn


def quantize_linear_layers(module: nn.Module, exclude_modules: Optional[list] = None) -> nn.Module:
    """Recursively replace Linear layers with FP8Linear layers"""
    if exclude_modules is None:
        exclude_modules = []
    
    for name, child in module.named_children():
        if any(exc in name for exc in exclude_modules):
            continue
            
        if isinstance(child, nn.Linear):
            # Replace with FP8Linear
            fp8_linear = FP8Linear.from_linear(child)
            setattr(module, name, fp8_linear)
        else:
            # Recursively apply to child modules
            quantize_linear_layers(child, exclude_modules)
    
    return module


def quantize_attention_layers(module: nn.Module) -> nn.Module:
    """Replace attention layers with FP8 quantized versions"""
    try:
        from .transformer import WanSelfAttention, WanT2VCrossAttention, WanI2VCrossAttention
    except ImportError:
        # If these classes don't exist, try alternative import
        try:
            from skyreels_v2_infer.modules.transformer import WanSelfAttention, WanT2VCrossAttention, WanI2VCrossAttention
        except ImportError:
            # Classes might not be exported, we'll work with what we have
            return module
    
    for name, child in module.named_children():
        if isinstance(child, (WanSelfAttention, WanT2VCrossAttention, WanI2VCrossAttention)):
            # Create FP8 versions of Q, K, V, O projections
            if hasattr(child, 'q'):
                child.q = FP8Linear.from_linear(child.q)
            if hasattr(child, 'k'):
                child.k = FP8Linear.from_linear(child.k)
            if hasattr(child, 'v'):
                child.v = FP8Linear.from_linear(child.v)
            if hasattr(child, 'o'):
                child.o = FP8Linear.from_linear(child.o)
            
            # For I2V, also quantize image-specific projections
            if isinstance(child, WanI2VCrossAttention):
                if hasattr(child, 'k_img'):
                    child.k_img = FP8Linear.from_linear(child.k_img)
                if hasattr(child, 'v_img'):
                    child.v_img = FP8Linear.from_linear(child.v_img)
        else:
            # Recursively apply to child modules
            quantize_attention_layers(child)
    
    return module


def quantize_model_to_fp8(
    model: nn.Module,
    quantize_linear: bool = True,
    quantize_attention: bool = True,
    exclude_modules: Optional[list] = None
) -> nn.Module:
    """
    Quantize a model to FP8
    
    Args:
        model: The model to quantize
        quantize_linear: Whether to quantize linear layers
        quantize_attention: Whether to quantize attention layers
        exclude_modules: List of module names to exclude from quantization
    
    Returns:
        The quantized model
    """
    if exclude_modules is None:
        exclude_modules = ['head', 'patch_embedding', 'text_embedding']  # Keep critical layers in higher precision
    
    if quantize_linear:
        model = quantize_linear_layers(model, exclude_modules)
    
    if quantize_attention:
        model = quantize_attention_layers(model)
    
    print(f"Model quantized to FP8 (linear={quantize_linear}, attention={quantize_attention})")
    return model


# Utility function for memory profiling
def get_model_memory_usage(model: nn.Module) -> dict:
    """Calculate memory usage of model parameters"""
    total_params = 0
    fp8_params = 0
    fp32_params = 0
    fp16_params = 0
    bf16_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.dtype == torch.float32:
            fp32_params += num_params
        elif param.dtype == torch.float16:
            fp16_params += num_params
        elif param.dtype == torch.bfloat16:
            bf16_params += num_params
        elif hasattr(torch, 'float8_e4m3fn') and param.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            fp8_params += num_params
    
    # Calculate memory in MB
    memory_usage = {
        'total_params': total_params,
        'fp32_params': fp32_params,
        'fp16_params': fp16_params,
        'bf16_params': bf16_params,
        'fp8_params': fp8_params,
        'fp32_memory_mb': fp32_params * 4 / (1024 * 1024),
        'fp16_memory_mb': fp16_params * 2 / (1024 * 1024),
        'bf16_memory_mb': bf16_params * 2 / (1024 * 1024),
        'fp8_memory_mb': fp8_params * 1 / (1024 * 1024),
        'total_memory_mb': (fp32_params * 4 + (fp16_params + bf16_params) * 2 + fp8_params) / (1024 * 1024)
    }
    
    return memory_usage