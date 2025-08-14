#!/bin/bash

# Script optimisé simple utilisant les fonctionnalités existantes
# Basé sur les découvertes du paper mais adapté au code actuel

set -e

# Configuration optimale
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:256'
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

echo -e "${MAGENTA}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║        SkyReels V2 - Optimisations Simplifiées            ║${NC}"
echo -e "${MAGENTA}╚══════════════════════════════════════════════════════════╝${NC}"

MODE=${1:-fast}
PROMPT="${2:-A beautiful cinematic scene}"

case $MODE in
    fast)
        echo -e "\n${GREEN}Mode FAST: Génération rapide avec TeaCache${NC}"
        NUM_FRAMES=25
        STEPS=12
        TEACACHE_THRESH=0.35
        ;;
    
    balanced)
        echo -e "\n${YELLOW}Mode BALANCED: Équilibré qualité/vitesse${NC}"
        NUM_FRAMES=33
        STEPS=18
        TEACACHE_THRESH=0.25
        ;;
    
    quality)
        echo -e "\n${CYAN}Mode QUALITY: Meilleure qualité${NC}"
        NUM_FRAMES=33
        STEPS=25
        TEACACHE_THRESH=0.15
        ;;
    
    *)
        echo "Modes: fast, balanced, quality"
        exit 1
        ;;
esac

echo -e "\n${CYAN}Configuration:${NC}"
echo -e "  Frames: $NUM_FRAMES"
echo -e "  Steps: $STEPS"
echo -e "  TeaCache: $TEACACHE_THRESH"
echo -e "  Prompt: \"$PROMPT\""

# Vérification GPU
echo -e "\n${GREEN}GPU Status:${NC}"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

echo ""
read -p "Lancer? (y/n) " -n 1 -r
echo
[[ ! $REPLY =~ ^[Yy]$ ]] && exit 0

# Lancement avec les bonnes options
echo -e "\n${GREEN}🚀 Lancement...${NC}"

python3 generate_video.py \
    --model_id /home/server/dev/SkyReels-V2-fork/checkpoints/14B-T2V-540P \
    --resolution 540P \
    --num_frames $NUM_FRAMES \
    --inference_steps $STEPS \
    --guidance_scale 6.0 \
    --shift 8.0 \
    --fps 24 \
    --seed 42 \
    --offload \
    --teacache \
    --teacache_thresh $TEACACHE_THRESH \
    --use_ret_steps \
    --prompt "$PROMPT"

echo -e "\n${GREEN}✅ Terminé!${NC}"