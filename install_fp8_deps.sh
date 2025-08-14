#!/bin/bash

echo "Installing optional FP8 dependencies for SkyReels V2..."
echo "=================================================="

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Warning: Not in a virtual environment. It's recommended to use a virtual environment."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install psutil and gputil for monitoring
echo "Installing monitoring tools..."
pip install psutil gputil

# Check PyTorch version and install if needed
echo "Checking PyTorch version..."
python -c "import torch; print(f'Current PyTorch version: {torch.__version__}')"

# Check if PyTorch version supports FP8
python -c "
import torch
import sys
version = torch.__version__.split('.')
major = int(version[0])
minor = int(version[1].split('+')[0]) if '+' in version[1] else int(version[1])
if major > 2 or (major == 2 and minor >= 3):
    print('PyTorch version supports native FP8')
    sys.exit(0)
else:
    print('PyTorch version does not support native FP8')
    sys.exit(1)
" 

if [ $? -ne 0 ]; then
    echo "Your PyTorch version doesn't support native FP8."
    echo "Consider upgrading to PyTorch 2.3+ for native FP8 support:"
    echo "  pip install torch>=2.3.0"
fi

# Try to install transformer-engine for NVIDIA GPUs
echo ""
echo "Attempting to install transformer-engine for optimized FP8..."
echo "This requires CUDA 11.8+ and may take a while..."

# Check CUDA version
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "CUDA version detected: $cuda_version"
    
    # Try to install transformer-engine
    pip install transformer-engine 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✓ transformer-engine installed successfully!"
    else
        echo "⚠ transformer-engine installation failed. FP8 will use fallback implementation."
        echo "  This is normal and FP8 will still work, just with slightly less optimization."
    fi
else
    echo "CUDA not found. Skipping transformer-engine installation."
    echo "FP8 will use fallback implementation."
fi

echo ""
echo "=================================================="
echo "FP8 dependencies installation complete!"
echo ""
echo "To test FP8 quantization, run:"
echo "  python test_fp8_quantization.py --model_path /path/to/model"
echo ""
echo "To use FP8 in generation, add --use_fp8 flag:"
echo "  python generate_video.py --use_fp8 --model_id /path/to/model --prompt 'your prompt'"
echo ""