# 🎬 Guide Comparatif - Modèle 14B vs Diffusion Forcing 1.3B

## 📊 Tableau Comparatif

| Caractéristique | Modèle 14B Standard | Diffusion Forcing 1.3B |
|-----------------|-------------------|------------------------|
| **Taille modèle** | 14 milliards params | 1.3 milliards params |
| **Mémoire GPU** | ~25-30GB | ~8-12GB |
| **Frames max (safe)** | 33-49 | 97-385+ |
| **Qualité** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Vitesse** | Lent | Rapide |
| **Stabilité longue durée** | ❌ OOM après 49 frames | ✅ Stable jusqu'à 385+ |

## 🚀 Commandes Recommandées

### Pour vidéos COURTES (< 2 secondes) - Utilisez le 14B
```bash
# 33 frames (~1.4 secondes) - Haute qualité
./run_single_gpu.sh normal
```

### Pour vidéos MOYENNES (2-4 secondes) - Utilisez Diffusion Forcing
```bash
# 97 frames (~4 secondes) - Génération directe
./run_df_long_video.sh 97 "Your amazing prompt"
```

### Pour vidéos LONGUES (4-8 secondes) - Diffusion Forcing ONLY
```bash
# 193 frames (~8 secondes) - Génération par segments
./run_df_long_video.sh 193 "Epic cinematic sequence"
```

### Pour vidéos TRÈS LONGUES (8-16 secondes) - Diffusion Forcing
```bash
# 385 frames (~16 secondes!) - Maximum pratique
./run_df_long_video.sh 385 "Long narrative scene"
```

## 🎯 Stratégie Optimale

### Workflow Recommandé

1. **Test rapide avec DF** (pour valider le prompt)
   ```bash
   ./run_df_long_video.sh 49 "Test prompt"
   ```

2. **Si besoin de qualité maximale ET < 33 frames**
   ```bash
   ./run_single_gpu.sh normal
   ```

3. **Pour production de vidéos longues**
   ```bash
   ./run_df_long_video.sh 193 "Final prompt"
   ```

## 💡 Avantages du Diffusion Forcing

### ✅ Points Forts
- **Génération par segments** : Peut créer des vidéos très longues
- **Overlap intelligent** : Maintient la cohérence entre segments
- **Économe en mémoire** : Modèle 1.3B = 10x moins de VRAM
- **Flexible** : Support image de début/fin, extension vidéo

### 📝 Paramètres Clés
- `--overlap_history` : Frames de chevauchement (17 ou 37)
- `--ar_step` : Pas autorégressif pour segments
- `--addnoise_condition` : Contrôle du bruit (20 recommandé)
- `--causal_attention` : Attention causale pour cohérence

## 🔧 Exemples Pratiques

### 1. Vidéo narrative longue (8 secondes)
```bash
./run_df_long_video.sh 193 \
  "A warrior's journey from village to mountain peak, epic adventure"
```

### 2. Boucle parfaite (4 secondes)
```bash
python3 generate_video_df.py \
  --model_id Skywork/SkyReels-V2-DF-1.3B-540P \
  --num_frames 97 \
  --image start.jpg \
  --end_image start.jpg \
  --prompt "Seamless loop of ocean waves"
```

### 3. Extension de vidéo existante
```bash
python3 generate_video_df.py \
  --video_path input.mp4 \
  --num_frames 193 \
  --overlap_history 37 \
  --prompt "Continue the scene"
```

## 📈 Benchmarks de Performance

### Sur RTX 5090

| Méthode | Frames | Temps | VRAM | Résultat |
|---------|--------|-------|------|----------|
| 14B Standard | 33 | ~2.5min | 25GB | ✅ OK |
| 14B Standard | 49 | ~4min | 30GB | ❌ OOM |
| DF 1.3B | 97 | ~3min | 12GB | ✅ OK |
| DF 1.3B | 193 | ~6min | 15GB | ✅ OK |
| DF 1.3B | 385 | ~12min | 18GB | ✅ OK |

## 🎨 Qualité vs Durée

```
Qualité maximale (14B):
├── 17 frames  ⭐⭐⭐⭐⭐
├── 33 frames  ⭐⭐⭐⭐⭐
└── 49 frames  ❌ OOM

Durée maximale (DF 1.3B):
├── 97 frames  ⭐⭐⭐⭐
├── 193 frames ⭐⭐⭐⭐
├── 289 frames ⭐⭐⭐½
└── 385 frames ⭐⭐⭐
```

## 🏆 Recommandations Finales

### Utilisez le modèle 14B pour :
- ✅ Clips très courts (<2 sec) nécessitant la qualité maximale
- ✅ Plans fixes ou avec peu de mouvement
- ✅ Génération de références haute qualité

### Utilisez Diffusion Forcing pour :
- ✅ Vidéos moyennes à longues (2-16 sec)
- ✅ Séquences narratives complexes
- ✅ Animations avec beaucoup de mouvement
- ✅ Production en volume (plus rapide)
- ✅ Extension/continuation de vidéos existantes

## 🚦 Quick Start

```bash
# Vidéo courte haute qualité (1.4 sec)
./run_single_gpu.sh normal

# Vidéo moyenne standard (4 sec)
./run_df_long_video.sh 97

# Vidéo longue narrative (8 sec)
./run_df_long_video.sh 193

# Challenge : vidéo très longue (16 sec!)
./run_df_long_video.sh 385
```

## 💬 Conclusion

**Pour 99% des cas, utilisez Diffusion Forcing** - c'est plus stable, plus rapide, et permet des vidéos beaucoup plus longues. Réservez le modèle 14B uniquement pour des cas spécifiques nécessitant la qualité absolue sur des clips très courts.