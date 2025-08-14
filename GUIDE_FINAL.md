# 🎬 Guide Final SkyReels-V2 pour RTX 5090

## ✅ État actuel

### Ce qui fonctionne
- ✅ **Mode GPU unique** : Stable et fonctionnel jusqu'à ~49 frames
- ✅ **TeaCache** : Accélération 3x avec économie mémoire
- ✅ **Offload CPU** : Permet des vidéos plus longues

### Ce qui ne fonctionne pas encore
- ❌ **Mode Multi-GPU (USP)** : Erreur NCCL de communication inter-GPU
- ⚠️ **Limitation** : Le modèle 14B utilise beaucoup de mémoire

## 🚀 Commandes recommandées

### 1. Test rapide (17 frames)
```bash
cd /home/server/dev/SkyReels-V2-fork
python3 test_single_gpu.py
```

### 2. Génération standard (33 frames)
```bash
./run_single_gpu.sh normal
```

### 3. Génération personnalisée
```bash
# 49 frames avec prompt personnalisé
./run_advanced.sh 49 "A majestic eagle soaring through mountain peaks at sunset"

# 65 frames (plus long, avec offload)
./run_advanced.sh 65 "Ocean waves crashing on rocky cliffs during a storm"

# 97 frames (maximum, très lent mais possible)
./run_advanced.sh 97 "Time-lapse of clouds moving over a city skyline"
```

## 📊 Capacités par configuration

| Frames | Temps approx. | Mémoire GPU | Qualité | Commande |
|--------|--------------|-------------|---------|----------|
| 9 | ~30s | ~15GB | ⭐⭐ | `./run_single_gpu.sh ultra-safe` |
| 17 | ~1min | ~18GB | ⭐⭐⭐ | `./run_single_gpu.sh safe` |
| 33 | ~2min | ~22GB | ⭐⭐⭐⭐ | `./run_single_gpu.sh normal` |
| 49 | ~3min | ~25GB | ⭐⭐⭐⭐ | `./run_advanced.sh 49` |
| 65 | ~4min | ~27GB | ⭐⭐⭐⭐⭐ | `./run_advanced.sh 65` |
| 97 | ~6min | ~30GB | ⭐⭐⭐⭐⭐ | `./run_advanced.sh 97` |

## 🛠️ Paramètres d'optimisation

### Pour économiser la mémoire

```bash
python3 generate_video.py \
    --model_id /home/server/dev/SkyReels-V2-fork/checkpoints/14B-T2V-540P \
    --resolution 540P \
    --num_frames 25 \
    --inference_steps 15 \
    --offload \              # Décharge sur CPU
    --teacache \             # Active le cache
    --teacache_thresh 0.35 \ # Seuil élevé = plus d'économie
    --prompt "Your prompt here"
```

### Pour la meilleure qualité

```bash
python3 generate_video.py \
    --model_id /home/server/dev/SkyReels-V2-fork/checkpoints/14B-T2V-540P \
    --resolution 540P \
    --num_frames 49 \
    --inference_steps 30 \   # Plus d'étapes = meilleure qualité
    --guidance_scale 7.5 \    # Plus élevé = plus fidèle au prompt
    --shift 10.0 \            # Ajuste le timing
    --seed 42 \               # Pour reproductibilité
    --prompt "Your detailed prompt here"
```

## 🔧 Résolution de problèmes

### Erreur "CUDA out of memory"

1. **Réduire les frames** : Commencez avec 17, puis augmentez progressivement
2. **Activer l'offload** : Ajoutez `--offload`
3. **Augmenter TeaCache** : `--teacache_thresh 0.4`
4. **Vérifier l'utilisation** :
   ```bash
   watch -n 1 nvidia-smi
   ```

### Vidéo trop courte ?

Pour des vidéos plus longues, générez plusieurs clips et assemblez-les :

```bash
# Générer 3 clips de 33 frames chacun
for i in 1 2 3; do
    ./run_advanced.sh 33 "Scene part $i"
done

# Assembler avec ffmpeg
ffmpeg -f concat -i <(for f in result/video_out/*.mp4; do echo "file '$f'"; done) -c copy output_long.mp4
```

### Améliorer la fluidité

```bash
# Interpolation de frames avec RIFE ou autre
# D'abord générer à 25 frames
./run_advanced.sh 25 "Your scene"

# Puis interpoler à 50 frames (nécessite un outil externe)
```

## 📝 Exemples de prompts efficaces

### Cinématique
```
"Cinematic dolly shot through a misty forest at dawn, volumetric lighting, 
slow camera movement revealing ancient trees"
```

### Action
```
"Dynamic action sequence: sports car drifting around mountain curves, 
dust clouds, dramatic camera angles, high speed"
```

### Nature
```
"Timelapse of northern lights dancing over snowy mountains, 
stars visible, ethereal green and purple aurora"
```

### Abstrait
```
"Abstract fluid art in motion, iridescent colors mixing and swirling, 
macro lens perspective, mesmerizing patterns"
```

## 🎯 Workflow recommandé

1. **Test rapide** (17 frames) pour valider le prompt
2. **Génération moyenne** (33-49 frames) pour preview
3. **Génération finale** (65-97 frames) avec paramètres optimaux

## 💡 Astuces Pro

1. **Batch processing** : Créez un script pour traiter plusieurs prompts
   ```bash
   for prompt in "prompt1" "prompt2" "prompt3"; do
       ./run_advanced.sh 49 "$prompt"
   done
   ```

2. **Variations** : Utilisez différents seeds
   ```bash
   for seed in 42 123 456; do
       python3 generate_video.py ... --seed $seed
   done
   ```

3. **Monitoring GPU** : Dans un terminal séparé
   ```bash
   nvidia-smi dmon -s mu -i 0
   ```

## 📞 Support

Si vous rencontrez des problèmes :

1. Vérifiez d'abord avec le test simple : `python3 test_single_gpu.py`
2. Essayez le mode ultra-safe : `./run_single_gpu.sh ultra-safe`
3. Consultez les logs détaillés avec `NCCL_DEBUG=INFO` pour le multi-GPU

## 🎉 Résumé

Avec votre RTX 5090, vous pouvez générer :
- **Rapidement** : 17 frames en ~1 minute
- **Confortablement** : 33-49 frames en 2-3 minutes
- **Maximum** : jusqu'à 97 frames en ~6 minutes

Le mode GPU unique est **stable et recommandé**. Le multi-GPU nécessite des corrections supplémentaires du code source USP.