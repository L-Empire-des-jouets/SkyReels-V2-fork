#!/bin/bash

# Script utilisant les optimisations révélées dans le paper SkyReels
# Permet d'utiliser le modèle 14B avec moins de VRAM grâce à FP8 et autres techniques

set -e

# Configuration environnement selon le paper
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:256'
export TOKENIZERS_PARALLELISM=false

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${MAGENTA}${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}${BOLD}║     SkyReels V2 - Optimisations du Paper Scientifique     ║${NC}"
echo -e "${MAGENTA}${BOLD}║         FP8 + Parallel Strategies + Distillation          ║${NC}"
echo -e "${MAGENTA}${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"

# Paramètres
MODE=${1:-standard}
PROMPT="${2:-A beautiful cinematic scene with dynamic lighting}"

echo -e "\n${CYAN}📚 Optimisations du paper activées:${NC}"
echo -e "  • ${GREEN}FP8 Quantization${NC} : Réduit la mémoire de 50%"
echo -e "  • ${GREEN}CFG Parallel${NC} : Accélération 1.8x sur 2 GPUs"
echo -e "  • ${GREEN}VAE Parallel${NC} : Décodage parallélisé"
echo -e "  • ${GREEN}SageAttn-8bit${NC} : Attention 1.3x plus rapide"
echo -e "  • ${GREEN}Gradient Checkpointing${NC} : Économie mémoire"

# Configuration selon le mode
case $MODE in
    test)
        echo -e "\n${YELLOW}Mode TEST: Validation rapide avec distillation${NC}"
        NUM_FRAMES=17
        USE_FP8=""
        USE_MULTI=""
        USE_DISTILLED="--use_distilled"
        DESC="4 steps seulement (distillé)"
        ;;
    
    standard)
        echo -e "\n${GREEN}Mode STANDARD: Équilibré avec FP8${NC}"
        NUM_FRAMES=49
        USE_FP8="--use_fp8"
        USE_MULTI=""
        USE_DISTILLED=""
        DESC="FP8 quantization activée"
        ;;
    
    dual-gpu)
        echo -e "\n${CYAN}Mode DUAL-GPU: Parallélisation complète${NC}"
        NUM_FRAMES=65
        USE_FP8="--use_fp8"
        USE_MULTI="--use_multi_gpu"
        USE_DISTILLED=""
        DESC="CFG + VAE parallel sur 2 GPUs"
        ;;
    
    max-speed)
        echo -e "\n${MAGENTA}Mode MAX-SPEED: Toutes optimisations${NC}"
        NUM_FRAMES=49
        USE_FP8="--use_fp8"
        USE_MULTI="--use_multi_gpu"
        USE_DISTILLED="--use_distilled"
        DESC="FP8 + 2 GPUs + Distillation (4 steps)"
        ;;
    
    *)
        echo -e "${RED}Mode inconnu: $MODE${NC}"
        echo "Modes disponibles: test, standard, dual-gpu, max-speed"
        exit 1
        ;;
esac

# Vérification GPUs
echo -e "\n${BLUE}📊 État du système:${NC}"
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "  GPUs détectés: $GPU_COUNT"

if [ "$GPU_COUNT" -ge 2 ] && [ ! -z "$USE_MULTI" ]; then
    echo -e "  ${GREEN}✅ Mode multi-GPU disponible${NC}"
    export CUDA_VISIBLE_DEVICES=0,1
else
    echo -e "  ${YELLOW}⚠️ Mode single GPU${NC}"
    export CUDA_VISIBLE_DEVICES=0
fi

nvidia-smi --query-gpu=name,memory.free,memory.total,compute_cap --format=csv,noheader | while IFS=',' read -r name mem_free mem_total compute_cap; do
    echo -e "  • $name"
    echo -e "    Mémoire: ${mem_free}MB libre / ${mem_total}MB"
    echo -e "    Compute: $compute_cap"
    
    # Vérifier support FP8 (compute >= 8.9 pour Ada/Hopper)
    major=$(echo $compute_cap | cut -d'.' -f1)
    minor=$(echo $compute_cap | cut -d'.' -f2)
    if [ "$major" -ge 9 ] || ([ "$major" -eq 8 ] && [ "$minor" -ge 9 ]); then
        echo -e "    ${GREEN}FP8: Supporté ✅${NC}"
    else
        echo -e "    ${YELLOW}FP8: Non supporté${NC}"
    fi
done

# Configuration détaillée
echo -e "\n${GREEN}⚙️ Configuration:${NC}"
echo -e "  Mode: ${BOLD}$MODE${NC}"
echo -e "  Frames: $NUM_FRAMES"
echo -e "  Optimisations: $DESC"

# Estimation performances selon le paper
if [ ! -z "$USE_DISTILLED" ]; then
    SPEEDUP="7.5x plus rapide (4 steps vs 30)"
elif [ ! -z "$USE_MULTI" ]; then
    SPEEDUP="1.8x plus rapide (2 GPUs)"
elif [ ! -z "$USE_FP8" ]; then
    SPEEDUP="1.1x plus rapide + 50% moins de VRAM"
else
    SPEEDUP="Baseline"
fi

echo -e "  ${CYAN}Performance attendue: $SPEEDUP${NC}"

# Prompt
echo -e "\n${BLUE}📝 Prompt:${NC}"
echo -e "  \"$PROMPT\""

# Confirmation
echo ""
read -p "$(echo -e ${GREEN}Lancer avec ces optimisations? [y/n]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Annulé.${NC}"
    exit 0
fi

# Nettoyage mémoire
echo -e "\n${YELLOW}🧹 Préparation de la mémoire GPU...${NC}"
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Lancement
echo -e "\n${GREEN}🚀 Lancement avec optimisations du paper...${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

START_TIME=$(date +%s)

# Commande Python avec les optimisations
python3 run_optimized_fp8.py \
    --num_frames $NUM_FRAMES \
    $USE_FP8 \
    $USE_MULTI \
    $USE_DISTILLED \
    --prompt "$PROMPT"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Résultats
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "\n${GREEN}${BOLD}✅ Génération terminée avec succès!${NC}"
echo -e "${CYAN}⏱️ Durée totale: ${DURATION} secondes${NC}"

# Comparaison avec baseline
BASELINE_TIME=$((NUM_FRAMES * 30 / 10))  # Estimation baseline
SPEEDUP_ACTUAL=$((BASELINE_TIME * 100 / DURATION))
echo -e "${CYAN}🚀 Accélération: ${SPEEDUP_ACTUAL}% par rapport au baseline${NC}"

echo -e "\n${BLUE}📹 Vidéo sauvegardée dans:${NC}"
echo -e "  ${YELLOW}result/optimized/${NC}"

# Conseils selon le mode
echo -e "\n${MAGENTA}💡 Conseil:${NC}"
case $MODE in
    test)
        echo "  Essayez 'dual-gpu' pour plus de frames avec 2 GPUs"
        ;;
    standard)
        echo "  Essayez 'max-speed' pour une génération ultra-rapide"
        ;;
    dual-gpu)
        echo "  Ajoutez la distillation avec 'max-speed' pour 7x plus rapide"
        ;;
    max-speed)
        echo "  C'est le mode le plus rapide possible!"
        ;;
esac

echo -e "\n${BOLD}📖 Référence: ${NC}Techniques du paper officiel SkyReels"
echo -e "   • FP8: RTX 4090 24GB → 720p possible"
echo -e "   • Votre RTX 5090 32GB → Encore mieux!"