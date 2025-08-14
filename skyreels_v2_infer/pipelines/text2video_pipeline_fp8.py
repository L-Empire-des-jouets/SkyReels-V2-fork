"""
FP8 Optimized Text2Video Pipeline for SkyReels-V2
Implements video generation with FP8 quantized models for reduced VRAM usage
"""

import os
from typing import List, Optional, Union
import numpy as np
import torch
import gc
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm

from ..modules import get_text_encoder, get_vae
from ..modules.transformer_fp8 import WanModelFP8
from ..scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler


class Text2VideoPipelineFP8:
    """
    FP8 Optimized Text2Video Pipeline
    Uses quantized models to reduce VRAM usage while maintaining quality
    """
    
    def __init__(
        self, 
        model_path, 
        dit_path=None, 
        device: str = "cuda", 
        weight_dtype=torch.bfloat16, 
        use_usp=False, 
        offload=False,
        use_fp8=True,
        use_sage_attention=True
    ):
        """
        Initialize the FP8 optimized pipeline
        
        Args:
            model_path: Path to the model checkpoint
            dit_path: Path to the DiT checkpoint (if separate)
            device: Device to run on
            weight_dtype: Data type for weights
            use_usp: Whether to use USP (Unified Sequence Parallel)
            offload: Whether to offload models to CPU when not in use
            use_fp8: Whether to use FP8 quantization
            use_sage_attention: Whether to use SageAttention for additional memory savings
        """
        self.device = device
        self.offload = offload
        self.use_fp8 = use_fp8
        self.use_sage_attention = use_sage_attention
        
        load_device = "cpu" if offload else device
        
        # Check if we have an FP8 checkpoint
        fp8_checkpoint_path = os.path.join(dit_path or model_path, "dit_checkpoint_fp8.pt")
        
        if os.path.exists(fp8_checkpoint_path) and use_fp8:
            print(f"Loading FP8 quantized model from {fp8_checkpoint_path}")
            self.transformer = self._load_fp8_model(fp8_checkpoint_path, load_device, weight_dtype)
        else:
            print(f"Loading standard model and converting to FP8...")
            # Load standard model and convert to FP8
            from ..modules import get_transformer
            standard_model = get_transformer(dit_path or model_path, load_device, weight_dtype)
            
            if use_fp8:
                # Convert to FP8
                from ..modules.fp8_quantization import quantize_linear_layers
                self.transformer = quantize_linear_layers(
                    standard_model,
                    exclude_layers=['head', 'embedding']
                )
                print("Model converted to FP8 quantization")
            else:
                self.transformer = standard_model
        
        # Load VAE (keep in FP32 for quality)
        vae_model_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        self.vae = get_vae(vae_model_path, device, weight_dtype=torch.float32)
        
        # Load text encoder (can be quantized later if needed)
        self.text_encoder = get_text_encoder(model_path, load_device, weight_dtype)
        
        self.video_processor = VideoProcessor(vae_scale_factor=16)
        self.sp_size = 1
        
        # Setup USP if requested
        if use_usp:
            self._setup_usp()
        
        self.scheduler = FlowUniPCMultistepScheduler()
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        
        # Print memory statistics
        self._print_memory_stats()
    
    def _load_fp8_model(self, checkpoint_path, device, dtype):
        """Load an FP8 quantized model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get configuration
        config = checkpoint.get('config', {})
        
        # Create FP8 model
        model = WanModelFP8(
            dim=config.get('dim', 3072),
            num_blocks=config.get('num_blocks', 32),
            num_heads=config.get('num_heads', 24),
            text_dim=config.get('text_dim', 4096),
            freq_dim=config.get('freq_dim', 256),
            use_fp8=True,
            use_sage_attention=self.use_sage_attention
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(dtype).to(device)
        
        print(f"Loaded FP8 model with memory reduction: {checkpoint.get('memory_stats', {}).get('memory_reduction', 0)*100:.1f}%")
        
        return model
    
    def _setup_usp(self):
        """Setup Unified Sequence Parallel"""
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward
        import types
        
        for block in self.transformer.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        
        self.transformer.forward = types.MethodType(usp_dit_forward, self.transformer)
        self.sp_size = get_sequence_parallel_world_size()
    
    def _print_memory_stats(self):
        """Print current memory usage statistics"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            if hasattr(self.transformer, 'estimate_memory_usage'):
                stats = self.transformer.estimate_memory_usage()
                print(f"Model Memory - Total: {stats['total_memory_gb']:.2f}GB, Reduction: {stats['memory_reduction']*100:.1f}%")
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        width: int = 544,
        height: int = 960,
        num_frames: int = 97,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        shift: float = 5.0,
        generator: Optional[torch.Generator] = None,
        enable_memory_efficient_attention: bool = True,
    ):
        """
        Generate video from text prompt with FP8 optimization
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            width: Video width
            height: Video height
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for CFG
            shift: Shift parameter for scheduler
            generator: Random generator for reproducibility
            enable_memory_efficient_attention: Whether to use memory efficient attention
        
        Returns:
            List of video frames as numpy arrays
        """
        # Preprocess
        F = num_frames
        target_shape = (
            self.vae.vae.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            height // self.vae_stride[1],
            width // self.vae_stride[2],
        )
        
        # Encode text with potential CPU offloading
        self.text_encoder.to(self.device)
        context = self.text_encoder.encode(prompt).to(self.device)
        context_null = self.text_encoder.encode(negative_prompt).to(self.device)
        
        if self.offload:
            self.text_encoder.cpu()
            torch.cuda.empty_cache()
        
        # Initialize latents
        latents = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            )
        ]
        
        # Move transformer to device
        self.transformer.to(self.device)
        
        # Use automatic mixed precision for additional memory savings
        with torch.cuda.amp.autocast(dtype=self.transformer.dtype), torch.no_grad():
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, shift=shift)
            timesteps = self.scheduler.timesteps
            
            # Enable memory efficient attention if requested
            if enable_memory_efficient_attention and hasattr(self.transformer, 'enable_xformers_memory_efficient_attention'):
                self.transformer.enable_xformers_memory_efficient_attention()
            
            # Denoising loop with progress bar
            for _, t in enumerate(tqdm(timesteps, desc="Generating video")):
                latent_model_input = torch.stack(latents)
                timestep = torch.stack([t])
                
                # Clear cache periodically to prevent fragmentation
                if _ % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Classifier-free guidance
                noise_pred_cond = self.transformer(latent_model_input, t=timestep, context=context)[0]
                noise_pred_uncond = self.transformer(latent_model_input, t=timestep, context=context_null)[0]
                
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # Scheduler step
                temp_x0 = self.scheduler.step(
                    noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0), 
                    return_dict=False, generator=generator
                )[0]
                latents = [temp_x0.squeeze(0)]
        
        # Offload transformer if needed
        if self.offload:
            self.transformer.cpu()
            torch.cuda.empty_cache()
        
        # Decode latents to video
        videos = self.vae.decode(latents[0])
        videos = (videos / 2 + 0.5).clamp(0, 1)
        
        # Convert to numpy arrays
        videos = [video for video in videos]
        videos = [video.permute(1, 2, 3, 0) * 255 for video in videos]
        videos = [video.cpu().numpy().astype(np.uint8) for video in videos]
        
        # Clean up
        gc.collect()
        torch.cuda.empty_cache()
        
        return videos
    
    def enable_model_cpu_offload(self):
        """Enable CPU offloading for all models"""
        self.offload = True
        self.text_encoder.cpu()
        self.transformer.cpu()
        torch.cuda.empty_cache()
    
    def disable_model_cpu_offload(self):
        """Disable CPU offloading"""
        self.offload = False
        self.text_encoder.to(self.device)
        self.transformer.to(self.device)


# Export the FP8 pipeline
__all__ = ['Text2VideoPipelineFP8']