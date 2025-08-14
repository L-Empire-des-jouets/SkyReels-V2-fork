#!/bin/bash
# Installation script for FP8 support in SkyReels-V2

echo "=========================================="
echo "SkyReels-V2 FP8 Installation Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found. Please run this script from the SkyReels-V2-fork directory.${NC}"
    exit 1
fi

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: No virtual environment detected.${NC}"
    echo "Looking for skyreels_venv..."
    
    if [ -d "skyreels_venv" ]; then
        echo "Found skyreels_venv. Activating..."
        source skyreels_venv/bin/activate
    else
        echo -e "${RED}Virtual environment not found. Please activate your virtual environment first.${NC}"
        echo "Example: source skyreels_venv/bin/activate"
        exit 1
    fi
fi

echo -e "${GREEN}Virtual environment active: $VIRTUAL_ENV${NC}"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python version: $PYTHON_VERSION"

# Check CUDA availability
echo -e "\n${YELLOW}Checking CUDA availability...${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}CUDA GPUs detected${NC}"
else
    echo -e "${YELLOW}Warning: CUDA not detected or nvidia-smi not available${NC}"
fi

# Install required packages if missing
echo -e "\n${YELLOW}Checking required packages...${NC}"

python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "PyTorch not found. Please install PyTorch first."
    echo "Visit: https://pytorch.org/get-started/locally/"
    exit 1
fi

# Check PyTorch version and CUDA support
echo -e "\n${YELLOW}PyTorch Configuration:${NC}"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Check FP8 support
echo -e "\n${YELLOW}Checking FP8 support...${NC}"
python -c "
import torch
has_fp8 = hasattr(torch, 'float8_e4m3fn')
if has_fp8:
    print('✅ FP8 data types are available')
else:
    print('⚠️  FP8 data types not available - will use fallback to FP16')
    print('    FP8 requires PyTorch 2.1+ and compatible GPU (Ada Lovelace or newer)')
"

# Create test script
echo -e "\n${YELLOW}Creating test script...${NC}"
cat > test_fp8_setup.py << 'EOF'
#!/usr/bin/env python
"""Test FP8 setup"""
import sys
sys.path.insert(0, '.')

print("\nTesting FP8 modules...")
try:
    from skyreels_v2_infer.modules.fp8_quantization import HAS_FP8
    print(f"✅ FP8 quantization module loaded")
    print(f"   FP8 support: {HAS_FP8}")
    
    from skyreels_v2_infer.modules.transformer_fp8 import WanModelFP8
    print(f"✅ FP8 transformer module loaded")
    
    from skyreels_v2_infer.pipelines.text2video_pipeline_fp8 import Text2VideoPipelineFP8
    print(f"✅ FP8 pipeline module loaded")
    
    print("\n✅ All FP8 modules loaded successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

# Run test
echo -e "\n${YELLOW}Running FP8 module test...${NC}"
python test_fp8_setup.py

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}=========================================="
    echo "FP8 Setup Complete!"
    echo "==========================================${NC}"
    echo ""
    echo "You can now use the FP8 optimized pipeline:"
    echo ""
    echo "1. Convert model to FP8 (one-time):"
    echo "   python convert_to_fp8.py --model_path /path/to/14B-540P --verify"
    echo ""
    echo "2. Generate videos with FP8:"
    echo "   python generate_video_fp8.py --model_path /path/to/14B-540P --use_fp8"
    echo ""
    echo "For more options, see: FP8_USAGE.md"
else
    echo -e "\n${RED}FP8 setup failed. Please check the errors above.${NC}"
    exit 1
fi

# Clean up
rm -f test_fp8_setup.py