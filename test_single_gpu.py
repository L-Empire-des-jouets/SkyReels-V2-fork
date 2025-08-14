#!/usr/bin/env python3
"""
Script de test pour SkyReels-V2 sur un seul GPU
Pour diagnostiquer les problèmes avant d'utiliser multi-GPU
"""

import os
import torch
import gc
import imageio
import time
from diffusers.utils import load_image
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Text2VideoPipeline

# Configuration mémoire optimisée
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Utiliser seulement GPU 0

def cleanup_memory():
    """Nettoie la mémoire GPU"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def main():
    print("=" * 60)
    print("Test SkyReels-V2 sur GPU unique")
    print("=" * 60)
    
    # Configuration
    model_path = "/home/server/dev/SkyReels-V2-fork/checkpoints/14B-T2V-540P"
    prompt = "A beautiful sunset over the ocean with gentle waves"
    
    # Paramètres ultra-safe pour test
    num_frames = 17  # Très peu de frames
    inference_steps = 10  # Peu d'étapes
    height = 544
    width = 960
    
    print(f"\nConfiguration de test:")
    print(f"  - Model: {model_path}")
    print(f"  - Frames: {num_frames}")
    print(f"  - Steps: {inference_steps}")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Device: cuda:0")
    
    # Vérifier la disponibilité GPU
    if not torch.cuda.is_available():
        print("❌ CUDA n'est pas disponible!")
        return
    
    print(f"\n✅ GPU détecté: {torch.cuda.get_device_name(0)}")
    print(f"   Mémoire totale: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Nettoyer la mémoire avant de commencer
    cleanup_memory()
    
    try:
        # Initialiser le pipeline (sans USP)
        print("\n📦 Chargement du modèle...")
        pipe = Text2VideoPipeline(
            model_path=model_path,
            dit_path=model_path,
            use_usp=False,  # Pas de parallélisation
            offload=True,   # Offload CPU pour économiser la mémoire
            weight_dtype=torch.float16,  # float16 pour économiser la mémoire
            device="cuda:0"
        )
        print("✅ Modèle chargé avec succès")
        
        # Activer TeaCache pour économiser la mémoire
        print("\n🔧 Activation de TeaCache...")
        pipe.transformer.initialize_teacache(
            enable_teacache=True,
            num_steps=inference_steps,
            teacache_thresh=0.35,  # Seuil élevé pour plus d'économie
            use_ret_steps=False,
            ckpt_dir=model_path
        )
        
        # Préparer les arguments
        kwargs = {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted",
            "num_frames": num_frames,
            "num_inference_steps": inference_steps,
            "guidance_scale": 6.0,
            "shift": 8.0,
            "generator": torch.Generator(device="cuda:0").manual_seed(42),
            "height": height,
            "width": width,
        }
        
        print(f"\n🎬 Génération de la vidéo...")
        print(f"   Prompt: {prompt}")
        
        # Nettoyer la mémoire avant génération
        cleanup_memory()
        
        # Générer la vidéo
        start_time = time.time()
        with torch.amp.autocast('cuda', dtype=torch.float16), torch.no_grad():
            video_frames = pipe(**kwargs)[0]
        
        generation_time = time.time() - start_time
        print(f"\n✅ Génération réussie en {generation_time:.1f} secondes!")
        
        # Sauvegarder la vidéo
        os.makedirs("result/test", exist_ok=True)
        output_path = "result/test/test_single_gpu.mp4"
        imageio.mimwrite(output_path, video_frames, fps=24, quality=8, output_params=["-loglevel", "error"])
        print(f"📹 Vidéo sauvegardée: {output_path}")
        
        # Afficher l'utilisation mémoire finale
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n📊 Utilisation mémoire GPU:")
        print(f"   Allouée: {mem_allocated:.2f} GB")
        print(f"   Réservée: {mem_reserved:.2f} GB")
        
        print("\n✅ Test terminé avec succès!")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ Erreur de mémoire GPU!")
        print("   Le modèle 14B est trop grand même avec ces paramètres minimaux.")
        print("\n💡 Solutions possibles:")
        print("   1. Utiliser le modèle 1.3B au lieu du 14B")
        print("   2. Réduire encore num_frames (essayez 9)")
        print("   3. Utiliser une résolution plus basse")
        raise e
    
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        raise e
    
    finally:
        # Nettoyer la mémoire
        cleanup_memory()

if __name__ == "__main__":
    main()