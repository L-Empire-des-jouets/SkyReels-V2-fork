#!/bin/bash

# Script de lancement pour SkyReels-V2 sur GPU unique (sans USP)
# Alternative plus stable sans parallélisation multi-GPU

set -e

# Configuration de l'environnement
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export CUDA_VISIBLE_DEVICES=0  # Utiliser seulement le GPU 0
export NCCL_DEBUG=INFO  # Pour debug si nécessaire

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== SkyReels-V2 - Mode GPU Unique (Plus Stable) ===${NC}"
echo -e "${YELLOW}⚠️  Note: Ce mode utilise un seul GPU mais est plus stable${NC}"

# Paramètres
MODEL_PATH="/home/server/dev/SkyReels-V2-fork/checkpoints/14B-T2V-540P"
MODE=${1:-safe}
PROMPT="${2:-A serene lake surrounded by mountains}"

# Configuration selon le mode
case $MODE in
    ultra-safe)
        echo -e "${RED}Mode ULTRA-SAFE: Minimaliste pour test${NC}"
        NUM_FRAMES=9
        INFERENCE_STEPS=8
        TEACACHE_THRESH=0.4
        ;;
    
    safe)
        echo -e "${YELLOW}Mode SAFE: Économie maximale de mémoire${NC}"
        NUM_FRAMES=17
        INFERENCE_STEPS=12
        TEACACHE_THRESH=0.35
        ;;
    
    normal)
        echo -e "${GREEN}Mode NORMAL: Équilibré${NC}"
        NUM_FRAMES=33
        INFERENCE_STEPS=20
        TEACACHE_THRESH=0.25
        ;;
    
    *)
        echo -e "${RED}Mode inconnu: $MODE${NC}"
        echo "Modes disponibles: ultra-safe, safe, normal"
        exit 1
        ;;
esac

# Affichage configuration
echo -e "\n${GREEN}Configuration:${NC}"
echo "  - Mode: $MODE"
echo "  - GPU: Single GPU (cuda:0)"
echo "  - Frames: $NUM_FRAMES"
echo "  - Steps: $INFERENCE_STEPS"
echo "  - Model: 14B-T2V-540P"
echo ""

# Vérifier l'état du GPU
echo -e "${BLUE}État du GPU:${NC}"
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits | head -1

echo ""
read -p "Lancer la génération? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Annulé."
    exit 0
fi

# Commande Python directe (pas de torchrun)
echo -e "\n${GREEN}Lancement sur GPU unique...${NC}"

python3 generate_video.py \
    --model_id "$MODEL_PATH" \
    --resolution 540P \
    --num_frames $NUM_FRAMES \
    --inference_steps $INFERENCE_STEPS \
    --guidance_scale 6.0 \
    --shift 8.0 \
    --fps 24 \
    --seed 42 \
    --offload \
    --teacache \
    --teacache_thresh $TEACACHE_THRESH \
    --prompt "$PROMPT"

echo -e "\n${GREEN}✅ Génération terminée!${NC}"