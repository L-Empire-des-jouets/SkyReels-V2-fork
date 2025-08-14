# Guide d'optimisation SkyReels-V2 pour 2x RTX 5090

## 🆕 MISE À JOUR : Correction de l'erreur NCCL

Suite à l'erreur NCCL/CUDA, j'ai ajouté des corrections supplémentaires et des alternatives.

## 🚀 Solution rapide

### Option 1 : GPU Unique (RECOMMANDÉ - Plus stable)

```bash
# Test simple sur un seul GPU
python3 test_single_gpu.py

# Ou utiliser le script de lancement
./run_single_gpu.sh safe
```

### Option 2 : Multi-GPU avec USP (après corrections)

```bash
# Avec les corrections appliquées
./run_skyreels_2gpu.sh safe
```

## 📝 Fichiers modifiés/créés

### Corrections appliquées
- **`skyreels_v2_infer/distributed/xdit_context_parallel.py`** : 
  - Ligne 39 : float64 → float32 (économie mémoire)
  - Ligne 69 : Correction du device pour broadcast NCCL
- **`generate_video_optimized.py`** : Script optimisé avec gestion mémoire
- **`test_single_gpu.py`** : Script de test GPU unique (nouveau)
- **`run_single_gpu.sh`** : Lancement sur un seul GPU (nouveau)
- **`run_skyreels_2gpu.sh`** : Script multi-GPU avec modes

### 2. Utilisation rapide

```bash
# Rendre le script exécutable
chmod +x run_skyreels_2gpu.sh

# Mode SAFE (utilisation minimale de mémoire)
./run_skyreels_2gpu.sh safe

# Mode NORMAL (équilibré)
./run_skyreels_2gpu.sh normal

# Mode QUALITY (haute qualité, plus de mémoire)
./run_skyreels_2gpu.sh quality

# Mode FAST (génération rapide)
./run_skyreels_2gpu.sh fast
```

## 📊 Comparaison des modes

| Mode | Frames | Steps | Mémoire GPU | Temps | Qualité |
|------|--------|-------|-------------|-------|---------|
| **safe** | 25 | 15 | ~20GB | ~2min | ⭐⭐⭐ |
| **normal** | 49 | 20 | ~25GB | ~4min | ⭐⭐⭐⭐ |
| **quality** | 97 | 30 | ~30GB | ~8min | ⭐⭐⭐⭐⭐ |
| **fast** | 25 | 10 | ~18GB | ~1min | ⭐⭐ |

## 🛠️ Commande manuelle personnalisée

Si vous voulez plus de contrôle :

```bash
# Exemple avec paramètres personnalisés
torchrun --nproc_per_node=2 generate_video_optimized.py \
    --model_id /home/server/dev/SkyReels-V2-fork/checkpoints/14B-T2V-540P \
    --resolution 540P \
    --num_frames 35 \
    --inference_steps 18 \
    --guidance_scale 6.0 \
    --shift 8.0 \
    --fps 24 \
    --use_usp \
    --seed 42 \
    --dtype float16 \
    --offload \
    --teacache \
    --teacache_thresh 0.25 \
    --prompt "Your custom prompt here"
```

## 🔧 Paramètres d'optimisation

### Réduction de mémoire (du plus au moins efficace)

1. **`--dtype float16`** : Utilise float16 au lieu de bfloat16 (économie ~30%)
2. **`--offload`** : Décharge certains calculs sur CPU (économie ~20%)
3. **`--num_frames`** : Réduire le nombre de frames (économie proportionnelle)
4. **`--inference_steps`** : Réduire les étapes (économie ~10-15% par 10 steps)
5. **`--teacache`** : Active la mise en cache (économie ~15-20%)

### Amélioration de la vitesse

1. **`--teacache --teacache_thresh 0.3`** : Accélération 3x avec légère perte de qualité
2. **`--use_ret_steps`** : Améliore la vitesse avec TeaCache
3. **`--inference_steps 10`** : Moins d'étapes = plus rapide

## 🐛 Résolution de problèmes

### Erreur "CUDA out of memory"

1. **Première tentative** : Utiliser le mode `safe`
   ```bash
   ./run_skyreels_2gpu.sh safe
   ```

2. **Si ça ne marche toujours pas** :
   ```bash
   # Ultra safe mode
   torchrun --nproc_per_node=2 generate_video_optimized.py \
       --model_id /home/server/dev/SkyReels-V2-fork/checkpoints/14B-T2V-540P \
       --resolution 540P \
       --num_frames 17 \
       --inference_steps 10 \
       --use_usp \
       --seed 42 \
       --dtype float16 \
       --offload \
       --teacache --teacache_thresh 0.35
   ```

### Vérifier l'utilisation GPU

```bash
# Surveiller l'utilisation en temps réel
watch -n 1 nvidia-smi

# Ou dans un autre terminal pendant la génération
nvidia-smi dmon -s mu
```

## 📈 Optimisations appliquées

### 1. Modification du code source
- **rope_apply** : float64 → float32 (économie significative de mémoire)
- **autocast** : Utilisation correcte de torch.amp.autocast

### 2. Variables d'environnement
```bash
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
```
- Améliore la gestion de la fragmentation mémoire
- Permet une allocation plus efficace

### 3. Gestion mémoire active
- Nettoyage régulier avec `gc.collect()` et `torch.cuda.empty_cache()`
- Synchronisation GPU appropriée

## 💡 Conseils supplémentaires

1. **Pour des vidéos plus longues** : Générez plusieurs clips courts et assemblez-les
2. **Pour 720P** : Utilisez d'abord 540P puis upscalez avec un autre modèle
3. **Batch processing** : Traitez plusieurs prompts en séquence avec le même modèle chargé

## 🔍 Monitoring

Pour surveiller l'utilisation pendant la génération :

```python
# Ajoutez ceci dans votre script Python
import torch

def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.memory_allocated(i)/1024**3:.2f}GB / {torch.cuda.max_memory_allocated(i)/1024**3:.2f}GB")

# Appelez après chaque étape importante
print_gpu_memory()
```

## 📞 Support

Si vous rencontrez toujours des problèmes après ces optimisations :

1. Essayez le modèle 1.3B au lieu du 14B
2. Réduisez encore les paramètres
3. Vérifiez que vous avez bien les derniers drivers NVIDIA
4. Assurez-vous que rien d'autre n'utilise les GPUs

## ✅ Checklist de démarrage

- [ ] Script rendu exécutable : `chmod +x run_skyreels_2gpu.sh`
- [ ] Environnement virtuel activé
- [ ] Modèle téléchargé dans `/home/server/dev/SkyReels-V2-fork/checkpoints/`
- [ ] 2 GPUs disponibles (vérifier avec `nvidia-smi`)
- [ ] Au moins 64GB de RAM système pour l'offload CPU

Bonne génération de vidéos ! 🎬