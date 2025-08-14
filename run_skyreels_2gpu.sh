#!/bin/bash

# Script de lancement optimisé pour SkyReels-V2 avec 2x RTX 5090
# Utilisation: ./run_skyreels_2gpu.sh [mode]
# Modes disponibles: safe, normal, quality, fast

set -e

# Configuration de l'environnement
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export CUDA_VISIBLE_DEVICES=0,1

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== SkyReels-V2 Launcher pour 2x RTX 5090 ===${NC}"

# Paramètres par défaut
MODEL_PATH="/home/server/dev/SkyReels-V2-fork/checkpoints/14B-T2V-540P"
PROMPT="A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface."
MODE=${1:-normal}

# Configuration selon le mode
case $MODE in
    safe)
        echo -e "${YELLOW}Mode SAFE: Utilisation minimale de mémoire${NC}"
        NUM_FRAMES=25
        INFERENCE_STEPS=15
        RESOLUTION="540P"
        USE_OFFLOAD="--offload"
        USE_TEACACHE="--teacache --teacache_thresh 0.3"
        DTYPE="float16"
        ;;
    
    normal)
        echo -e "${GREEN}Mode NORMAL: Équilibre qualité/performance${NC}"
        NUM_FRAMES=49
        INFERENCE_STEPS=20
        RESOLUTION="540P"
        USE_OFFLOAD=""
        USE_TEACACHE="--teacache --teacache_thresh 0.2"
        DTYPE="bfloat16"
        ;;
    
    quality)
        echo -e "${YELLOW}Mode QUALITY: Haute qualité${NC}"
        NUM_FRAMES=97
        INFERENCE_STEPS=30
        RESOLUTION="540P"
        USE_OFFLOAD=""
        USE_TEACACHE=""
        DTYPE="bfloat16"
        ;;
    
    fast)
        echo -e "${YELLOW}Mode FAST: Génération rapide${NC}"
        NUM_FRAMES=25
        INFERENCE_STEPS=10
        RESOLUTION="540P"
        USE_OFFLOAD=""
        USE_TEACACHE="--teacache --teacache_thresh 0.3 --use_ret_steps"
        DTYPE="float16"
        ;;
    
    *)
        echo -e "${RED}Mode inconnu: $MODE${NC}"
        echo "Modes disponibles: safe, normal, quality, fast"
        exit 1
        ;;
esac

# Affichage de la configuration
echo -e "\n${GREEN}Configuration:${NC}"
echo "  - Mode: $MODE"
echo "  - Frames: $NUM_FRAMES"
echo "  - Steps: $INFERENCE_STEPS"
echo "  - Resolution: $RESOLUTION"
echo "  - Data type: $DTYPE"
echo "  - Model: $MODEL_PATH"
echo ""

# Demander confirmation
read -p "Lancer la génération? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Annulé."
    exit 0
fi

# Commande de lancement avec USP (multi-GPU)
echo -e "\n${GREEN}Lancement avec USP (2 GPUs)...${NC}"

torchrun --nproc_per_node=2 generate_video_optimized.py \
    --model_id "$MODEL_PATH" \
    --resolution "$RESOLUTION" \
    --num_frames $NUM_FRAMES \
    --inference_steps $INFERENCE_STEPS \
    --guidance_scale 6.0 \
    --shift 8.0 \
    --fps 24 \
    --use_usp \
    --seed 42 \
    --dtype "$DTYPE" \
    $USE_OFFLOAD \
    $USE_TEACACHE \
    --prompt "$PROMPT"

echo -e "\n${GREEN}✅ Génération terminée!${NC}"