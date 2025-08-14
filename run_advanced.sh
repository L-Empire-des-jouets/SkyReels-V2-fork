#!/bin/bash

# Script avancé pour SkyReels-V2 - Génération de vidéos longues sur GPU unique
# Usage: ./run_advanced.sh [frames] [prompt]

set -e

# Configuration de l'environnement
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   SkyReels-V2 Advanced - Single GPU Mode      ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════╝${NC}"

# Paramètres
MODEL_PATH="/home/server/dev/SkyReels-V2-fork/checkpoints/14B-T2V-540P"
NUM_FRAMES=${1:-49}
PROMPT="${2:-A beautiful cinematic scene with dynamic camera movement}"

# Configuration automatique basée sur le nombre de frames
if [ $NUM_FRAMES -le 17 ]; then
    echo -e "${GREEN}Mode: RAPIDE (≤17 frames)${NC}"
    INFERENCE_STEPS=12
    TEACACHE_THRESH=0.35
    USE_OFFLOAD=""
elif [ $NUM_FRAMES -le 33 ]; then
    echo -e "${YELLOW}Mode: STANDARD (18-33 frames)${NC}"
    INFERENCE_STEPS=18
    TEACACHE_THRESH=0.30
    USE_OFFLOAD=""
elif [ $NUM_FRAMES -le 49 ]; then
    echo -e "${YELLOW}Mode: ÉTENDU (34-49 frames)${NC}"
    INFERENCE_STEPS=20
    TEACACHE_THRESH=0.25
    USE_OFFLOAD="--offload"
elif [ $NUM_FRAMES -le 65 ]; then
    echo -e "${RED}Mode: LONG (50-65 frames)${NC}"
    INFERENCE_STEPS=22
    TEACACHE_THRESH=0.20
    USE_OFFLOAD="--offload"
else
    echo -e "${RED}Mode: TRÈS LONG (>65 frames)${NC}"
    INFERENCE_STEPS=25
    TEACACHE_THRESH=0.15
    USE_OFFLOAD="--offload"
fi

# Vérification GPU
echo -e "\n${BLUE}📊 État du système:${NC}"
nvidia-smi --query-gpu=name,memory.free,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r name mem_free mem_total temp; do
    mem_used=$((mem_total - mem_free))
    mem_percent=$((mem_used * 100 / mem_total))
    echo -e "  GPU: $name"
    echo -e "  Mémoire: ${mem_used}MB / ${mem_total}MB (${mem_percent}% utilisé)"
    echo -e "  Température: ${temp}°C"
done

# Configuration détaillée
echo -e "\n${GREEN}⚙️  Configuration:${NC}"
echo -e "  ${CYAN}Frames:${NC} $NUM_FRAMES"
echo -e "  ${CYAN}Steps:${NC} $INFERENCE_STEPS"
echo -e "  ${CYAN}TeaCache:${NC} $TEACACHE_THRESH"
echo -e "  ${CYAN}Offload:${NC} $([ -z "$USE_OFFLOAD" ] && echo "Non" || echo "Oui")"
echo -e "  ${CYAN}Résolution:${NC} 960x544 (540P)"

# Estimation du temps
TIME_ESTIMATE=$((NUM_FRAMES * INFERENCE_STEPS / 10))
echo -e "\n${YELLOW}⏱️  Temps estimé: ~${TIME_ESTIMATE} secondes${NC}"

# Affichage du prompt
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

# Nettoyage mémoire avant lancement
echo -e "\n${YELLOW}🧹 Nettoyage de la mémoire GPU...${NC}"
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Lancement
echo -e "\n${GREEN}🚀 Lancement de la génération...${NC}"
echo "----------------------------------------"

START_TIME=$(date +%s)

python3 generate_video.py \
    --model_id "$MODEL_PATH" \
    --resolution 540P \
    --num_frames $NUM_FRAMES \
    --inference_steps $INFERENCE_STEPS \
    --guidance_scale 6.0 \
    --shift 8.0 \
    --fps 24 \
    --seed 42 \
    --teacache \
    --teacache_thresh $TEACACHE_THRESH \
    $USE_OFFLOAD \
    --prompt "$PROMPT"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Affichage final
echo "----------------------------------------"
echo -e "\n${GREEN}✅ Génération terminée avec succès!${NC}"
echo -e "${CYAN}⏱️  Durée totale: ${DURATION} secondes${NC}"

# Afficher l'emplacement de la vidéo
echo -e "\n${BLUE}📹 Vidéo sauvegardée dans:${NC}"
echo -e "  ${YELLOW}result/video_out/${NC}"

# Stats finales GPU
echo -e "\n${BLUE}📊 État final du GPU:${NC}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r used total; do
    percent=$((used * 100 / total))
    echo -e "  Mémoire utilisée: ${used}MB / ${total}MB (${percent}%)"
done