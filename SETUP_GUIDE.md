# SkyReels-V2 FP8 Setup Guide

## ⚠️ Important: Model Download Required

The model files are not present in your `checkpoints` directory. You need to download them first from Hugging Face.

## Step 1: Download the Model

First, you need to download the 14B model from Hugging Face:

```bash
# Activate your virtual environment
source skyreels_venv/bin/activate

# Download the 14B-540P model (recommended for 2x RTX 5090)
python download_models.py --model Skywork/SkyReels-V2-T2V-14B-540P

# Or download the 14B-720P model (requires more VRAM)
python download_models.py --model Skywork/SkyReels-V2-T2V-14B-720P
```

**Note**: The 14B model is approximately **30-40GB** in size. Make sure you have enough disk space and a stable internet connection.

## Step 2: Verify Download

After downloading, verify the model files:

```bash
ls -lah checkpoints/14B-540P/
```

You should see:
- Multiple `.safetensors` files (model weights)
- `config.json` (model configuration)
- `Wan2.1_VAE.pth` (VAE model)
- Text encoder files

## Step 3: Convert to FP8 (Optional but Recommended)

Once the model is downloaded, convert it to FP8 format to save VRAM:

```bash
python convert_to_fp8.py \
    --model_path checkpoints/14B-540P \
    --output_path checkpoints/14B-540P_fp8 \
    --verify
```

This will:
- Load the full 14B model
- Quantize linear layers to FP8
- Save the quantized model
- Show memory reduction statistics

Expected output:
```
Memory Statistics:
  Total Parameters: 14,000,000,000+
  FP8 Quantized Layers: 300+
  Original Memory (FP32): ~52 GB
  Quantized Memory (FP8): ~13 GB
  Memory Reduction: ~75%
```

## Step 4: Generate Videos

### Option A: With FP8 (Recommended)

```bash
python generate_video_fp8.py \
    --model_path checkpoints/14B-540P \
    --prompt "A beautiful sunset over the ocean" \
    --use_fp8 \
    --use_sage_attention \
    --offload
```

### Option B: Without FP8 (Original)

```bash
python generate_video.py \
    --model_id Skywork/SkyReels-V2-T2V-14B-540P \
    --prompt "A beautiful sunset over the ocean" \
    --offload
```

## Troubleshooting

### "Model path not found" Error

This means the model hasn't been downloaded yet. Run:
```bash
python download_models.py --model Skywork/SkyReels-V2-T2V-14B-540P
```

### "CUDA Out of Memory" Error

Even with FP8, if you get OOM errors:

1. **Enable CPU offloading**:
   ```bash
   python generate_video_fp8.py --offload
   ```

2. **Reduce resolution**:
   ```bash
   python generate_video_fp8.py --resolution 540P --num_frames 49
   ```

3. **Use both GPUs with USP**:
   ```bash
   torchrun --nproc_per_node=2 generate_video_fp8.py --use_usp
   ```

### "0% memory reduction" in FP8 conversion

This happens when the model weights aren't fully loaded. Make sure:
1. The model is completely downloaded
2. All `.safetensors` files are present
3. The model path is correct

## Model Sizes and Requirements

| Model | Download Size | FP32 Memory | FP8 Memory | Min VRAM |
|-------|--------------|-------------|------------|----------|
| 14B-540P | ~35GB | ~52GB | ~13GB | 16GB (with offload) |
| 14B-720P | ~35GB | ~52GB | ~13GB | 20GB (with offload) |

## Your Setup (2x RTX 5090)

With your 2x RTX 5090 (48GB total VRAM), you should be able to:

- ✅ Run 14B-540P with FP8 easily
- ✅ Run 14B-720P with FP8 and offloading
- ✅ Use both GPUs with USP for faster generation
- ✅ Generate 97 frames at 540P resolution
- ⚠️ May need offloading for 720P with 97 frames

## Quick Start Commands

```bash
# 1. Activate environment
source skyreels_venv/bin/activate

# 2. Download model (one-time, ~35GB)
python download_models.py

# 3. Convert to FP8 (one-time, saves 75% VRAM)
python convert_to_fp8.py --model_path checkpoints/14B-540P

# 4. Generate video with FP8
python generate_video_fp8.py \
    --model_path checkpoints/14B-540P \
    --prompt "A magical forest" \
    --use_fp8 \
    --offload

# 5. Multi-GPU generation (faster)
torchrun --nproc_per_node=2 generate_video_fp8.py \
    --model_path checkpoints/14B-540P \
    --use_fp8 \
    --use_usp
```

## Notes

- First model download will take 30-60 minutes depending on internet speed
- FP8 conversion takes ~5-10 minutes
- Video generation: ~2-3 minutes for 97 frames at 540P with FP8
- The FP8 model can be reused - you only need to convert once

## Support

If you encounter issues:
1. Check that all dependencies are installed: `pip install -r requirements.txt`
2. Verify CUDA is working: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check available VRAM: `nvidia-smi`
4. Review the detailed logs for specific error messages