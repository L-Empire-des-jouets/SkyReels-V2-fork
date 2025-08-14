#!/bin/bash

# Script pour générer des vidéos LONGUES avec Diffusion Forcing
# Utilise le modèle 1.3B plus léger et génère par segments

set -e

# Configuration environnement
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

echo -e "${MAGENTA}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║   SkyReels-V2 Diffusion Forcing - Vidéos Longues      ║${NC}"
echo -e "${MAGENTA}║            Modèle 1.3B optimisé pour la durée         ║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════════════╝${NC}"

# Paramètres
MODEL_PATH="/home/server/dev/SkyReels-V2-fork/checkpoints/1.3B-540P"
NUM_FRAMES=${1:-97}
PROMPT="${2:-A cinematic journey through a futuristic city at night}"

# Configuration selon le nombre de frames
if [ $NUM_FRAMES -le 97 ]; then
    echo -e "${GREEN}Mode: STANDARD (≤97 frames, génération directe)${NC}"
    OVERLAP_HISTORY=""
    BASE_FRAMES=97
    AR_STEP=0
    INFERENCE_STEPS=30
    TEACACHE_THRESH=0.2
elif [ $NUM_FRAMES -le 193 ]; then
    echo -e "${YELLOW}Mode: LONG (98-193 frames, 2 segments)${NC}"
    OVERLAP_HISTORY="--overlap_history 37"
    BASE_FRAMES=97
    AR_STEP=15
    INFERENCE_STEPS=25
    TEACACHE_THRESH=0.25
elif [ $NUM_FRAMES -le 289 ]; then
    echo -e "${YELLOW}Mode: TRÈS LONG (194-289 frames, 3 segments)${NC}"
    OVERLAP_HISTORY="--overlap_history 37"
    BASE_FRAMES=97
    AR_STEP=12
    INFERENCE_STEPS=20
    TEACACHE_THRESH=0.3
else
    echo -e "${RED}Mode: ULTRA LONG (>289 frames, 4+ segments)${NC}"
    OVERLAP_HISTORY="--overlap_history 17"
    BASE_FRAMES=97
    AR_STEP=10
    INFERENCE_STEPS=18
    TEACACHE_THRESH=0.35
fi

# Calcul durée vidéo
DURATION=$((NUM_FRAMES * 100 / 24 / 100))
echo -e "\n${CYAN}📹 Durée de la vidéo: ~${DURATION} secondes à 24 FPS${NC}"

# État GPU
echo -e "\n${BLUE}📊 État du système:${NC}"
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits | while IFS=',' read -r name mem_free mem_total; do
    mem_used=$((mem_total - mem_free))
    mem_percent=$((mem_used * 100 / mem_total))
    echo -e "  GPU: $name"
    echo -e "  Mémoire disponible: ${mem_free}MB / ${mem_total}MB"
done

# Configuration
echo -e "\n${GREEN}⚙️  Configuration Diffusion Forcing:${NC}"
echo -e "  ${CYAN}Modèle:${NC} 1.3B (léger, optimisé pour vidéos longues)"
echo -e "  ${CYAN}Frames totales:${NC} $NUM_FRAMES"
echo -e "  ${CYAN}Base frames:${NC} $BASE_FRAMES"
if [ ! -z "$OVERLAP_HISTORY" ]; then
    echo -e "  ${CYAN}Overlap:${NC} ${OVERLAP_HISTORY#*--overlap_history }"
    echo -e "  ${CYAN}AR steps:${NC} $AR_STEP"
fi
echo -e "  ${CYAN}Inference steps:${NC} $INFERENCE_STEPS"
echo -e "  ${CYAN}TeaCache:${NC} $TEACACHE_THRESH"

# Estimation temps
if [ $NUM_FRAMES -le 97 ]; then
    TIME_ESTIMATE=$((NUM_FRAMES * INFERENCE_STEPS / 15))
else
    TIME_ESTIMATE=$((NUM_FRAMES * INFERENCE_STEPS / 10))
fi
echo -e "\n${YELLOW}⏱️  Temps estimé: ~${TIME_ESTIMATE} secondes${NC}"

# Prompt
echo -e "\n${BLUE}📝 Prompt:${NC}"
echo -e "  \"$PROMPT\""

# Confirmation
echo ""
read -p "$(echo -e ${GREEN}Lancer la génération? [y/n]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Annulé.${NC}"
    exit 0
fi

# Nettoyage mémoire
echo -e "\n${YELLOW}🧹 Nettoyage de la mémoire GPU...${NC}"
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Commande de génération
echo -e "\n${GREEN}🚀 Lancement de la génération Diffusion Forcing...${NC}"
echo "----------------------------------------"

START_TIME=$(date +%s)

# Construction de la commande
CMD="python3 generate_video_df.py \
    --model_id $MODEL_PATH \
    --resolution 540P \
    --num_frames $NUM_FRAMES \
    --base_num_frames $BASE_FRAMES \
    --inference_steps $INFERENCE_STEPS \
    --guidance_scale 6.0 \
    --shift 8.0 \
    --fps 24 \
    --seed 42 \
    --offload \
    --teacache \
    --teacache_thresh $TEACACHE_THRESH"

# Ajouter les paramètres conditionnels
if [ ! -z "$OVERLAP_HISTORY" ]; then
    CMD="$CMD $OVERLAP_HISTORY"
fi

if [ $AR_STEP -gt 0 ]; then
    CMD="$CMD --ar_step $AR_STEP"
    CMD="$CMD --causal_attention --causal_block_size 1"
    CMD="$CMD --addnoise_condition 20"
fi

# Ajouter le prompt
CMD="$CMD --prompt \"$PROMPT\""

# Exécuter
eval $CMD

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Résultats
echo "----------------------------------------"
echo -e "\n${GREEN}✅ Génération terminée avec succès!${NC}"
echo -e "${CYAN}⏱️  Durée totale: ${DURATION} secondes${NC}"
echo -e "${CYAN}📊 Frames générées: $NUM_FRAMES${NC}"

# Emplacement vidéo
echo -e "\n${BLUE}📹 Vidéo sauvegardée dans:${NC}"
echo -e "  ${YELLOW}result/diffusion_forcing/${NC}"

# Stats GPU finales
echo -e "\n${BLUE}📊 État final du GPU:${NC}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r used total; do
    percent=$((used * 100 / total))
    echo -e "  Mémoire utilisée: ${used}MB / ${total}MB (${percent}%)"
done

echo -e "\n${MAGENTA}💡 Astuce: Le Diffusion Forcing peut générer jusqu'à 385+ frames!${NC}"
echo -e "${MAGENTA}   Essayez: ./run_df_long_video.sh 193 \"Your prompt\"${NC}"