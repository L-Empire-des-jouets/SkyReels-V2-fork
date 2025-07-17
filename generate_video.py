import os
import torch
import deepspeed

# Active un logging plus verbeux pour NCCL (optionnel, diagnostic)
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

# Configuration pour optimiser l'utilisation mémoire
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    # ——————— Pinning et init du groupe distribué NCCL ———————
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="gloo")
    # ——————————————————————————————————————————————————————

    import argparse
    import gc
    import random
    import time
    import imageio
    from diffusers.utils import load_image

    from skyreels_v2_infer.modules import download_model
    from skyreels_v2_infer.pipelines import (
        Text2VideoPipeline,
        Image2VideoPipeline,
        PromptEnhancer,
        resizecrop,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="video_out")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Skywork/SkyReels-V2-T2V-14B-540P",
    )
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"], required=True)
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--use_usp", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "A serene lake surrounded by towering mountains, with a few swans "
            "gracefully gliding across the water and sunlight dancing on the surface."
        ),
    )
    parser.add_argument("--prompt_enhancer", action="store_true")
    parser.add_argument("--teacache", action="store_true")
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="Higher speedup with cache at cost of some quality (0.1=2x,0.2=3x)",
    )
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        help="Enable retention steps for faster generation and better quality",
    )
    # Nouveaux arguments pour l'optimisation mémoire
    parser.add_argument("--cpu_offload_aggressive", action="store_true", help="Aggressive CPU offloading")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--sequential_cpu_offload", action="store_true", help="Enable sequential CPU offload")
    
    args = parser.parse_args()

    # Download ou validation du modèle
    args.model_id = download_model(args.model_id)
    print("model_id:", args.model_id)

    # Gestion du seed
    assert (not args.use_usp) or (args.use_usp and args.seed is not None), \
        "--use_usp requires --seed"
    if args.seed is None:
        random.seed(time.time())
        args.seed = random.randrange(2**32 - 1)

    # Configuration résolution
    if args.resolution == "540P":
        height, width = 544, 960
    else:
        height, width = 720, 1280

    # Nettoyage mémoire préventif
    gc.collect()
    torch.cuda.empty_cache()

    # Chargement optionnel d'une image
    image = load_image(args.image).convert("RGB") if args.image else None

    # Negative prompt
    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
        "paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, "
        "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
        "disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, "
        "many people in the background, walking backwards"
    )

    # Distributed USP setup (optionnel)
    if args.use_usp:
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        import torch.distributed as dist

        # 1) Init CPU <-> CPU (pour la broadcast du RNG) via Gloo
        deepspeed.init_distributed(dist_backend="gloo")

        # 2) Pinning du process sur la bonne GPU
        torch.cuda.set_device(local_rank)

        # 3) Récupère le nombre total de ranks
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # 4) Initialise l'environnement USP (xfuser)
        init_distributed_environment(rank=local_rank, world_size=world_size)
        initialize_model_parallel(
            sequence_parallel_degree=world_size,
            ring_degree=1,
            ulysses_degree=world_size,
        )

    # Prompt enhancer
    prompt_input = args.prompt
    if args.prompt_enhancer and image is None:
        prompt_input = PromptEnhancer()(prompt_input)
        print(f"enhanced prompt: {prompt_input}")
        gc.collect()
        torch.cuda.empty_cache()

    # Initialisation du pipeline avec optimisations mémoire
    if image is None:
        print("init text2video pipeline")
        pipe = Text2VideoPipeline(
            model_path=args.model_id,
            dit_path=args.model_id,
            use_usp=args.use_usp,
            offload=args.offload,
        )
        # Évite de déplacer le text_encoder sur GPU pour économiser de la mémoire
        def _noop_to(*args, **kwargs):
            return pipe.text_encoder
        pipe.text_encoder.to = _noop_to
        
        # Déplace le text_encoder sur CPU de manière permanente
        pipe.text_encoder = pipe.text_encoder.to("cpu")
        
    else:
        print("init image2video pipeline")
        pipe = Image2VideoPipeline(
            model_path=args.model_id,
            dit_path=args.model_id,
            use_usp=args.use_usp,
            offload=args.offload,
        )
        image = resizecrop(image, height, width)
        # Déplace le text_encoder sur CPU pour I2V aussi
        pipe.text_encoder = pipe.text_encoder.to("cpu")

    # Optimisations mémoire supplémentaires
    if args.sequential_cpu_offload:
        # Offload séquentiel des composants
        if hasattr(pipe, 'vae'):
            pipe.vae = pipe.vae.to("cpu")
        if hasattr(pipe, 'unet'):
            pipe.unet = pipe.unet.to("cpu")
        print("Sequential CPU offload enabled")

    if args.gradient_checkpointing:
        # Active le gradient checkpointing si disponible
        try:
            if hasattr(pipe.transformer, 'enable_gradient_checkpointing'):
                pipe.transformer.enable_gradient_checkpointing()
                print("Gradient checkpointing enabled")
            elif hasattr(pipe.transformer, '_set_gradient_checkpointing'):
                # Pour WanModel, essaie la méthode directe
                pipe.transformer._set_gradient_checkpointing(True)
                print("Gradient checkpointing enabled (WanModel)")
        except Exception as e:
            print(f"Gradient checkpointing not supported: {e}")
            print("Continuing without gradient checkpointing")

    # Configuration DeepSpeed pour l'offload
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Configuration ZeRO compatible avec init_inference
    zero_config = {
        "offload_param": {
            "device": "cpu",
            "nvme_path": "offload",
            "pin_memory": False,
        }
    }

    if args.cpu_offload_aggressive:
        # Configuration plus aggressive pour l'offload
        zero_config["offload_param"]["pin_memory"] = True
        print("Aggressive CPU offload enabled")

    # Wrapping the transformer with DeepSpeed Inference Engine
    engine = deepspeed.init_inference(
        pipe.transformer,
        tensor_parallel={"tp_size": world_size},
        dtype=torch.float16,
        replace_with_kernel_inject=True,
        zero=zero_config,
    )
    pipe.transformer = engine

    # Nettoyage mémoire après init DeepSpeed
    gc.collect()
    torch.cuda.empty_cache()

    # Teacache (optionnel)
    if args.teacache:
        engine.module.initialize_teacache(
            enable_teacache=True,
            num_steps=args.inference_steps,
            teacache_thresh=args.teacache_thresh,
            use_ret_steps=args.use_ret_steps,
            ckpt_dir=args.model_id,
        )

    # Fonction pour déplacer les tenseurs sur GPU temporairement
    def move_to_gpu_temporarily(model, device):
        """Déplace le modèle sur GPU temporairement pour l'inférence"""
        if hasattr(model, 'to'):
            return model.to(device)
        return model

    def move_to_cpu_after_use(model):
        """Remet le modèle sur CPU après utilisation"""
        if hasattr(model, 'to'):
            model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    # Prépare les kwargs pour l'inférence
    kwargs = {
        "prompt": prompt_input,
        "negative_prompt": negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "height": height,
        "width": width,
    }
    if image is not None:
        kwargs["image"] = image

    # Répertoire de sortie
    output_dir = os.path.join("result", args.outdir)
    os.makedirs(output_dir, exist_ok=True)

    # Monitoring mémoire
    def print_memory_usage():
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, "
                  f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")

    print("Memory usage before inference:")
    print_memory_usage()

    # Lancement de l'inférence avec gestion mémoire optimisée
    try:
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
            # Nettoyage préventif
            gc.collect()
            torch.cuda.empty_cache()
            
            print("Starting inference...")
            video_frames = pipe(**kwargs)[0]
            
            print("Memory usage after inference:")
            print_memory_usage()
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM Error: {e}")
        print("Trying with more aggressive memory management...")
        
        # Nettoyage complet
        gc.collect()
        torch.cuda.empty_cache()
        
        # Réduit la taille des frames si nécessaire
        if args.num_frames > 49:
            print("Reducing num_frames to 49 for memory efficiency")
            kwargs["num_frames"] = 49
            
        # Réduit les steps d'inférence
        if args.inference_steps > 20:
            print("Reducing inference_steps to 20 for memory efficiency")
            kwargs["num_inference_steps"] = 20
            
        # Retry avec paramètres réduits
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
            video_frames = pipe(**kwargs)[0]

    # Sauvegarde sur le master
    if local_rank == 0:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        safe_prompt = args.prompt[:50].replace('/', '_')
        filename = f"{safe_prompt}_{args.seed}_{timestamp}.mp4"
        path = os.path.join(output_dir, filename)
        imageio.mimwrite(
            path,
            video_frames,
            fps=args.fps,
            quality=8,
            output_params=["-loglevel", "error"],
        )
        print(f"Video saved to: {path}")
        
    # Nettoyage final
    gc.collect()
    torch.cuda.empty_cache()