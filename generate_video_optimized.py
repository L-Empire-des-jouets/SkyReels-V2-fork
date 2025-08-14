#!/usr/bin/env python3
"""
Script optimisé pour générer des vidéos avec SkyReels-V2 sur 2x RTX 5090
Gestion mémoire améliorée et options de configuration
"""

import argparse
import gc
import os
import random
import time
import torch

import imageio
from diffusers.utils import load_image

from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Image2VideoPipeline
from skyreels_v2_infer.pipelines import PromptEnhancer
from skyreels_v2_infer.pipelines import resizecrop
from skyreels_v2_infer.pipelines import Text2VideoPipeline

# Configuration pour économiser la mémoire
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

MODEL_ID_CONFIG = {
    "text2video": [
        "Skywork/SkyReels-V2-T2V-14B-540P",
        "Skywork/SkyReels-V2-T2V-14B-720P",
    ],
    "image2video": [
        "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "Skywork/SkyReels-V2-I2V-14B-540P",
        "Skywork/SkyReels-V2-I2V-14B-720P",
    ],
}


def cleanup_memory():
    """Nettoie la mémoire GPU"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Génération vidéo optimisée pour 2x RTX 5090")
    parser.add_argument("--outdir", type=str, default="video_out")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-T2V-14B-540P")
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"], default="540P")
    parser.add_argument("--num_frames", type=int, default=49, help="Nombre de frames (réduit pour économiser la mémoire)")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--inference_steps", type=int, default=20, help="Nombre d'étapes d'inférence (réduit pour économiser la mémoire)")
    parser.add_argument("--use_usp", action="store_true", help="Utiliser USP pour la parallélisation")
    parser.add_argument("--offload", action="store_true", help="Activer l'offload CPU pour économiser la mémoire GPU")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface.",
    )
    parser.add_argument("--prompt_enhancer", action="store_true")
    parser.add_argument("--teacache", action="store_true", help="Utiliser TeaCache pour accélérer la génération")
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="Seuil TeaCache - 0.1 pour 2.0x speedup, 0.2 pour 3.0x speedup"
    )
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        help="Utiliser Retention Steps pour une génération plus rapide et de meilleure qualité"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Type de données pour le modèle (float16 économise de la mémoire)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Taille du batch (réduire si OOM)"
    )
    args = parser.parse_args()

    # Télécharger/vérifier le modèle
    args.model_id = download_model(args.model_id)
    print(f"Model ID: {args.model_id}")
    
    # Configuration de la mémoire
    print("Configuration mémoire GPU optimisée activée")
    cleanup_memory()

    # Vérification et configuration du seed
    assert (args.use_usp and args.seed is not None) or (not args.use_usp), "Le mode USP nécessite un seed"
    if args.seed is None:
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))
    
    print(f"Seed utilisé: {args.seed}")

    # Configuration de la résolution
    if args.resolution == "540P":
        height = 544
        width = 960
    elif args.resolution == "720P":
        height = 720
        width = 1280
    else:
        raise ValueError(f"Résolution invalide: {args.resolution}")
    
    print(f"Résolution: {width}x{height}")

    # Chargement de l'image si fournie
    image = load_image(args.image).convert("RGB") if args.image else None
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    local_rank = 0
    
    # Configuration USP pour multi-GPU
    if args.use_usp:
        assert not args.prompt_enhancer, "`--prompt_enhancer` n'est pas autorisé avec `--use_usp`"
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        import torch.distributed as dist

        print("Initialisation du processus distribué pour USP...")
        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(dist.get_rank())
        device = "cuda"

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        print(f"USP configuré avec {dist.get_world_size()} GPUs")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configuration du dtype
    weight_dtype = torch.float16 if args.dtype == "float16" else (
        torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    )
    print(f"Utilisation du dtype: {args.dtype}")

    # Amélioration du prompt si demandée
    prompt_input = args.prompt
    if args.prompt_enhancer and args.image is None:
        print("Initialisation du prompt enhancer...")
        prompt_enhancer = PromptEnhancer()
        prompt_input = prompt_enhancer(prompt_input)
        print(f"Prompt amélioré: {prompt_input}")
        del prompt_enhancer
        cleanup_memory()

    # Initialisation du pipeline
    if image is None:
        assert "T2V" in args.model_id, f"Vérifiez le model_id: {args.model_id}"
        print("Initialisation du pipeline text2video...")
        print(f"  - Offload: {args.offload}")
        print(f"  - USP: {args.use_usp}")
        print(f"  - Weight dtype: {weight_dtype}")
        
        pipe = Text2VideoPipeline(
            model_path=args.model_id, 
            dit_path=args.model_id, 
            use_usp=args.use_usp, 
            offload=args.offload,
            weight_dtype=weight_dtype,
            device=device
        )
    else:
        assert "I2V" in args.model_id, f"Vérifiez le model_id: {args.model_id}"
        print("Initialisation du pipeline image2video...")
        pipe = Image2VideoPipeline(
            model_path=args.model_id, 
            dit_path=args.model_id, 
            use_usp=args.use_usp, 
            offload=args.offload,
            weight_dtype=weight_dtype,
            device=device
        )
        args.image = load_image(args.image)
        image_width, image_height = args.image.size
        if image_height > image_width:
            height, width = width, height
        args.image = resizecrop(args.image, height, width)

    # Configuration TeaCache si activée
    if args.teacache:
        print(f"Activation de TeaCache avec seuil {args.teacache_thresh}")
        pipe.transformer.initialize_teacache(
            enable_teacache=True, 
            num_steps=args.inference_steps,
            teacache_thresh=args.teacache_thresh, 
            use_ret_steps=args.use_ret_steps,
            ckpt_dir=args.model_id
        )

    # Préparation des arguments pour la génération
    kwargs = {
        "prompt": prompt_input,
        "negative_prompt": negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "generator": torch.Generator(device=device).manual_seed(args.seed),
        "height": height,
        "width": width,
    }

    if image is not None:
        kwargs["image"] = args.image.convert("RGB")

    # Création du dossier de sortie
    save_dir = os.path.join("result", args.outdir)
    os.makedirs(save_dir, exist_ok=True)

    # Génération de la vidéo avec gestion mémoire
    print("\nDébut de la génération vidéo...")
    print(f"Paramètres: {kwargs}")
    
    try:
        # Nettoyage mémoire avant génération
        cleanup_memory()
        
        # Utilisation du contexte autocast approprié
        with torch.amp.autocast('cuda', dtype=weight_dtype), torch.no_grad():
            video_frames = pipe(**kwargs)[0]
        
        print("Génération terminée avec succès!")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n⚠️  Erreur de mémoire GPU détectée!")
        print("Solutions recommandées:")
        print("1. Réduire --num_frames (actuellement: {})".format(args.num_frames))
        print("2. Réduire --inference_steps (actuellement: {})".format(args.inference_steps))
        print("3. Utiliser --offload pour décharger sur CPU")
        print("4. Utiliser --dtype float16 pour économiser la mémoire")
        print("5. Activer --teacache pour accélérer et économiser la mémoire")
        raise e
    
    finally:
        # Nettoyage final de la mémoire
        cleanup_memory()

    # Sauvegarde de la vidéo (seulement sur le rang 0 si USP)
    if local_rank == 0:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        video_out_file = f"{args.prompt[:100].replace('/','')}_{args.seed}_{current_time}.mp4"
        output_path = os.path.join(save_dir, video_out_file)
        
        print(f"\nSauvegarde de la vidéo: {output_path}")
        imageio.mimwrite(output_path, video_frames, fps=args.fps, quality=8, output_params=["-loglevel", "error"])
        print(f"✅ Vidéo sauvegardée avec succès!")
        print(f"   - Résolution: {width}x{height}")
        print(f"   - Frames: {args.num_frames}")
        print(f"   - FPS: {args.fps}")