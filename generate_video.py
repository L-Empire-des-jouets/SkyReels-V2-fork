import os
import torch
import deepspeed

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
    else:
        print("init image2video pipeline")
        pipe = Image2VideoPipeline(
            model_path=args.model_id,
            dit_path=args.model_id,
            use_usp=args.use_usp,
            offload=args.offload,
        )
        image = resizecrop(image, height, width)

    # FIX: Gérer le device du VAE de manière cohérente
    current_device = f"cuda:{local_rank}"
    
    # FIX: Wrapper VAE CORRIGÉ pour éviter la récursion infinie
    class VAEWrapper:
        def __init__(self, original_vae, device):
            # IMPORTANT: Utiliser object.__setattr__ pour éviter la récursion
            object.__setattr__(self, '_original_vae', original_vae)
            object.__setattr__(self, '_device', device)
            object.__setattr__(self, '_cpu_offload', args.sequential_cpu_offload)
            object.__setattr__(self, '_is_on_gpu', False)
            
            # Détecte si le VAE a une structure imbriquée
            if hasattr(original_vae, 'vae'):
                object.__setattr__(self, '_inner_vae', original_vae.vae)
                object.__setattr__(self, '_has_nested_structure', True)
            else:
                object.__setattr__(self, '_inner_vae', original_vae)
                object.__setattr__(self, '_has_nested_structure', False)
            
        @property
        def vae(self):
            """Retourne le VAE interne pour préserver la structure vae.vae"""
            if self._has_nested_structure:
                return self._inner_vae
            else:
                # Pour les VAE sans structure imbriquée, retourne self pour maintenir l'interface
                return self
            
        @vae.setter
        def vae(self, value):
            """Setter pour le VAE interne"""
            object.__setattr__(self, '_inner_vae', value)
            
        def _ensure_on_device(self, target_device):
            """Assure que le VAE est sur le bon device avec gestion mémoire"""
            if not self._cpu_offload:
                return
                
            try:
                current_device = next(self._inner_vae.parameters()).device
                if str(current_device) != str(target_device):
                    try:
                        self._inner_vae = self._inner_vae.to(target_device)
                        object.__setattr__(self, '_is_on_gpu', target_device != 'cpu')
                    except torch.cuda.OutOfMemoryError:
                        # Nettoie la mémoire et réessaie
                        gc.collect()
                        torch.cuda.empty_cache()
                        self._inner_vae = self._inner_vae.to(target_device)
                        object.__setattr__(self, '_is_on_gpu', target_device != 'cpu')
            except StopIteration:
                # Le modèle n'a pas de paramètres, pas besoin de le déplacer
                pass
            
        def _offload_if_needed(self):
            """Remet sur CPU si l'offload est activé"""
            if self._cpu_offload and self._is_on_gpu:
                try:
                    self._inner_vae = self._inner_vae.to("cpu")
                    object.__setattr__(self, '_is_on_gpu', False)
                    torch.cuda.empty_cache()
                except:
                    pass  # Ignore les erreurs de déplacement
            
        def encode(self, *args, **kwargs):
            self._ensure_on_device(self._device)
            
            # S'assurer que tous les inputs sont sur le bon device
            args = [arg.to(self._device) if hasattr(arg, 'to') else arg for arg in args]
            for k, v in kwargs.items():
                if hasattr(v, 'to'):
                    kwargs[k] = v.to(self._device)
            
            result = self._inner_vae.encode(*args, **kwargs)
            self._offload_if_needed()
            return result
            
        def decode(self, *args, **kwargs):
            self._ensure_on_device(self._device)
            
            # S'assurer que tous les inputs sont sur le bon device
            args = [arg.to(self._device) if hasattr(arg, 'to') else arg for arg in args]
            for k, v in kwargs.items():
                if hasattr(v, 'to'):
                    kwargs[k] = v.to(self._device)
            
            # Gestion spéciale pour WanVAE qui nécessite un paramètre 'scale'
            try:
                result = self._inner_vae.decode(*args, **kwargs)
            except TypeError as e:
                if "missing 1 required positional argument: 'scale'" in str(e):
                    # Pour WanVAE, utiliser le scale existant de l'objet parent (WanVAE wrapper)
                    # qui contient les bonnes valeurs mean/std
                    if 'scale' not in kwargs:
                        # Chercher le scale dans l'objet parent WanVAE
                        parent_vae = self._original_vae  # L'objet WanVAE complet
                        if hasattr(parent_vae, 'scale'):
                            # Utiliser le scale existant du WanVAE
                            scale = parent_vae.scale
                            # S'assurer que les tensors sont sur le bon device
                            if isinstance(scale, list) and len(scale) == 2:
                                scale_mean = scale[0].to(self._device) if hasattr(scale[0], 'to') else torch.tensor(scale[0], device=self._device)
                                scale_std_inv = scale[1].to(self._device) if hasattr(scale[1], 'to') else torch.tensor(scale[1], device=self._device)
                                kwargs['scale'] = [scale_mean, scale_std_inv]
                            else:
                                kwargs['scale'] = scale
                        else:
                            # Fallback: utiliser les valeurs par défaut de WanVAE (z_dim=16)
                            mean = torch.tensor([
                                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
                            ], device=self._device)
                            
                            std = torch.tensor([
                                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
                            ], device=self._device)
                            
                            std_inv = 1.0 / std
                            kwargs['scale'] = [mean, std_inv]
                    
                    result = self._inner_vae.decode(*args, **kwargs)
                else:
                    # Re-raise l'erreur si ce n'est pas le problème de scale
                    raise e
            
            self._offload_if_needed()
            return result
        
        def to(self, device):
            """Handle device placement calls - ne fait rien si offload est activé"""
            if not self._cpu_offload:
                self._inner_vae = self._inner_vae.to(device)
            return self
        
        def __getattr__(self, name):
            # Évite la récursion en utilisant object.__getattribute__ pour les attributs privés
            if name.startswith('_'):
                return object.__getattribute__(self, name)
            # Forward all other attribute access to the inner VAE
            return getattr(self._inner_vae, name)
        
        def __setattr__(self, name, value):
            # Handle special wrapper attributes avec object.__setattr__
            if name.startswith('_') or name in ['_original_vae', '_inner_vae', '_device', '_cpu_offload', '_has_nested_structure', '_is_on_gpu']:
                object.__setattr__(self, name, value)
            else:
                # Forward attribute setting to inner VAE
                setattr(self._inner_vae, name, value)

    # FIX: Wrapper similaire pour le text encoder CORRIGÉ
    class TextEncoderWrapper:
        def __init__(self, text_encoder, device):
            object.__setattr__(self, '_text_encoder', text_encoder)
            object.__setattr__(self, '_device', device)
            object.__setattr__(self, '_cpu_offload', args.sequential_cpu_offload)
            object.__setattr__(self, '_is_on_gpu', False)
            
        def _ensure_on_device(self, target_device):
            """Assure que le text encoder est sur le bon device avec gestion mémoire"""
            if not self._cpu_offload:
                return
                
            try:
                current_device = next(self._text_encoder.parameters()).device
                if str(current_device) != str(target_device):
                    try:
                        self._text_encoder = self._text_encoder.to(target_device)
                        object.__setattr__(self, '_is_on_gpu', target_device != 'cpu')
                    except torch.cuda.OutOfMemoryError:
                        # Nettoie la mémoire et réessaie
                        gc.collect()
                        torch.cuda.empty_cache()
                        self._text_encoder = self._text_encoder.to(target_device)
                        object.__setattr__(self, '_is_on_gpu', target_device != 'cpu')
            except StopIteration:
                # Le modèle n'a pas de paramètres
                pass
            
        def _offload_if_needed(self):
            """Remet sur CPU si l'offload est activé"""
            if self._cpu_offload and self._is_on_gpu:
                try:
                    self._text_encoder = self._text_encoder.to("cpu")
                    object.__setattr__(self, '_is_on_gpu', False)
                    torch.cuda.empty_cache()
                except:
                    pass
            
        def __call__(self, *args, **kwargs):
            self._ensure_on_device(self._device)
            
            # S'assurer que tous les inputs sont sur le bon device
            args = [arg.to(self._device) if hasattr(arg, 'to') else arg for arg in args]
            for k, v in kwargs.items():
                if hasattr(v, 'to'):
                    kwargs[k] = v.to(self._device)
            
            result = self._text_encoder(*args, **kwargs)
            self._offload_if_needed()
            return result
        
        def to(self, device):
            """Handle device placement calls - avec gestion d'offload"""
            if self._cpu_offload:
                # Si l'offload est activé, on ne fait rien ici
                # Le déplacement se fera dynamiquement lors de l'utilisation
                return self
            else:
                # Mode normal
                try:
                    self._text_encoder = self._text_encoder.to(device)
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    self._text_encoder = self._text_encoder.to(device)
                return self
        
        def __getattr__(self, name):
            # Évite la récursion pour les attributs privés
            if name.startswith('_'):
                return object.__getattribute__(self, name)
            # Forward all other attribute access to the wrapped text encoder
            return getattr(self._text_encoder, name)
        
        def __setattr__(self, name, value):
            # Handle special wrapper attributes
            if name.startswith('_') or name in ['_text_encoder', '_device', '_cpu_offload', '_is_on_gpu']:
                object.__setattr__(self, name, value)
            else:
                # Forward attribute setting to wrapped text encoder
                setattr(self._text_encoder, name, value)

    # Wrapping des composants avec gestion des devices
    if hasattr(pipe, 'vae'):
        pipe.vae = VAEWrapper(pipe.vae, current_device)
    
    if hasattr(pipe, 'text_encoder'):
        pipe.text_encoder = TextEncoderWrapper(pipe.text_encoder, current_device)

    # Optimisations mémoire avec device management
    if args.sequential_cpu_offload:
        # Déplace les composants sur CPU, mais garde les wrappers
        if hasattr(pipe, 'vae'):
            try:
                pipe.vae._inner_vae = pipe.vae._inner_vae.to("cpu")
            except Exception:
                pass
        if hasattr(pipe, 'text_encoder'):
            try:
                pipe.text_encoder._text_encoder = pipe.text_encoder._text_encoder.to("cpu")
            except Exception:
                pass
        
        # Nettoie la mémoire GPU après avoir déplacé les composants
        gc.collect()
        torch.cuda.empty_cache()
        print("Sequential CPU offload enabled")

    if args.gradient_checkpointing:
        # Active le gradient checkpointing si disponible
        try:
            if hasattr(pipe.transformer, 'enable_gradient_checkpointing'):
                pipe.transformer.enable_gradient_checkpointing()
            elif hasattr(pipe.transformer, '_set_gradient_checkpointing'):
                pipe.transformer._set_gradient_checkpointing(True)
        except Exception:
            pass

    if args.cpu_offload_aggressive:
        pass  # Configuration déjà appliquée dans zero_config

    # Configuration DeepSpeed pour l'offload
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Configuration ZeRO compatible avec init_inference
    zero_config = {
        "offload_param": {
            "device": "cpu",
            "nvme_path": "offload",
            "pin_memory": args.cpu_offload_aggressive,
        }
    }

    # Wrapping the transformer with DeepSpeed Inference Engine
    print("Initializing DeepSpeed inference engine...")
    
    # FIX: Configuration DeepSpeed plus robuste
    try:
        engine = deepspeed.init_inference(
            pipe.transformer,
            tensor_parallel={"tp_size": world_size},
            dtype=torch.float16,  # Utilise float16 pour la cohérence
            replace_with_kernel_inject=True,
            zero=zero_config,
        )
        pipe.transformer = engine
    except Exception as e:
        print(f"Warning: DeepSpeed initialization failed, continuing without optimization")

    # Nettoyage mémoire après init DeepSpeed
    gc.collect()
    torch.cuda.empty_cache()

    # Teacache (optionnel)
    if args.teacache:
        print("using teacache")
        try:
            if hasattr(pipe.transformer, 'module'):
                # DeepSpeed wrapper
                pipe.transformer.module.initialize_teacache(
                    enable_teacache=True,
                    num_steps=args.inference_steps,
                    teacache_thresh=args.teacache_thresh,
                    use_ret_steps=args.use_ret_steps,
                    ckpt_dir=args.model_id,
                )
            else:
                # Direct transformer
                pipe.transformer.initialize_teacache(
                    enable_teacache=True,
                    num_steps=args.inference_steps,
                    teacache_thresh=args.teacache_thresh,
                    use_ret_steps=args.use_ret_steps,
                    ckpt_dir=args.model_id,
                )
        except Exception:
            pass

    # Prépare les kwargs pour l'inférence
    kwargs = {
        "prompt": prompt_input,
        "negative_prompt": negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "generator": torch.Generator(device=current_device).manual_seed(args.seed),
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

    # FIX: Patch de la fonction usp_attn_forward avant l'inférence
    def patch_attention_function():
        """Patche la fonction d'attention pour corriger le problème de dtype"""
        try:
            import skyreels_v2_infer.distributed.xdit_context_parallel as xdit_module
            
            # Sauvegarde la fonction originale
            original_usp_attn_forward = xdit_module.usp_attn_forward
            
            def fixed_usp_attn_forward(self, x, grid_sizes, freqs, block_mask):
                """Version corrigée de usp_attn_forward"""
                b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
                half_dtypes = (torch.float16, torch.bfloat16)

                def half(x):
                    return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

                # query, key, value function
                def qkv_fn(x):
                    q = self.norm_q(self.q(x)).view(b, s, n, d)
                    k = self.norm_k(self.k(x)).view(b, s, n, d)
                    v = self.v(x).view(b, s, n, d)
                    return q, k, v

                x = x.to(self.q.weight.dtype)
                q, k, v = qkv_fn(x)

                if not self._flag_ar_attention:
                    q = xdit_module.rope_apply(q, grid_sizes, freqs)
                    k = xdit_module.rope_apply(k, grid_sizes, freqs)
                    
                    # FIX: S'assurer que tous les tenseurs ont le même dtype AVANT l'attention
                    target_dtype = torch.bfloat16 if q.dtype in half_dtypes else torch.float16
                    q = q.to(target_dtype)
                    k = k.to(target_dtype)
                    v = v.to(target_dtype)
                    
                else:
                    q = xdit_module.rope_apply(q, grid_sizes, freqs)
                    k = xdit_module.rope_apply(k, grid_sizes, freqs)
                    
                    # FIX: S'assurer que tous les tenseurs ont le même dtype
                    target_dtype = torch.bfloat16
                    q = q.to(target_dtype)
                    k = k.to(target_dtype)
                    v = v.to(target_dtype)
                    
                    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                        x = (
                            torch.nn.functional.scaled_dot_product_attention(
                                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=block_mask
                            )
                            .transpose(1, 2)
                            .contiguous()
                        )
                    # Retourner directement x si on utilise l'attention causale
                    x = x.flatten(2)
                    x = self.o(x)
                    return x

                # FIX: Pour xFuserLongContextAttention, s'assurer que q, k, v ont le même dtype
                # et utiliser la fonction half() de manière cohérente
                q_half = half(q)
                k_half = half(k) 
                v_half = half(v)
                
                # Vérification finale des dtypes
                assert q_half.dtype == k_half.dtype == v_half.dtype, \
                    f"Dtype mismatch: q={q_half.dtype}, k={k_half.dtype}, v={v_half.dtype}"
                
                from xfuser.core.long_ctx_attention import xFuserLongContextAttention
                x = xFuserLongContextAttention()(None, query=q_half, key=k_half, value=v_half, window_size=self.window_size)

                # output
                x = x.flatten(2)
                x = self.o(x)
                return x
            
            # Applique le patch
            xdit_module.usp_attn_forward = fixed_usp_attn_forward
            
        except Exception:
            pass
    
    # Applique le patch
    patch_attention_function()

    # Lancement de l'inférence avec gestion mémoire optimisée
    try:
        # Nettoyage préventif maximal avant inférence
        gc.collect()
        torch.cuda.empty_cache()
        
        # Réduction préventive des paramètres si mémoire critique
        if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.85:
            print("Critical memory usage detected - reducing parameters preemptively")
            if args.num_frames > 25:
                kwargs["num_frames"] = 25
                print(f"Reduced num_frames to {kwargs['num_frames']}")
            if args.inference_steps > 15:
                kwargs["num_inference_steps"] = 15
                print(f"Reduced inference_steps to {kwargs['num_inference_steps']}")
        
        with torch.amp.autocast('cuda', dtype=torch.float16), torch.no_grad():
            print("Starting inference...")
            video_frames = pipe(**kwargs)[0]
            
            print("Memory usage after inference:")
            print_memory_usage()
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM Error: {e}")
        print("Trying with more aggressive memory management...")
        
        # Nettoyage complet et plus agressif
        gc.collect()
        torch.cuda.empty_cache()
        
        # Force le déplacement de tous les composants sur CPU temporairement
        if hasattr(pipe, 'vae') and hasattr(pipe.vae, '_inner_vae'):
            pipe.vae._inner_vae = pipe.vae._inner_vae.to("cpu")
        if hasattr(pipe, 'text_encoder') and hasattr(pipe.text_encoder, '_text_encoder'):
            pipe.text_encoder._text_encoder = pipe.text_encoder._text_encoder.to("cpu")
        
        # Nettoyage après déplacement
        gc.collect()
        torch.cuda.empty_cache()
        
        # Réduction drastique des paramètres
        kwargs["num_frames"] = min(kwargs["num_frames"], 16)
        kwargs["num_inference_steps"] = min(kwargs["num_inference_steps"], 10)
        print(f"Emergency mode: num_frames={kwargs['num_frames']}, inference_steps={kwargs['num_inference_steps']}")
        
        # Retry avec paramètres ultra-conservateurs
        try:
            with torch.amp.autocast('cuda', dtype=torch.float16), torch.no_grad():
                video_frames = pipe(**kwargs)[0]
        except torch.cuda.OutOfMemoryError:
            # Dernière tentative avec paramètres minimaux
            kwargs["num_frames"] = 8
            kwargs["num_inference_steps"] = 5
            print(f"Final attempt: num_frames={kwargs['num_frames']}, inference_steps={kwargs['num_inference_steps']}")
            
            with torch.amp.autocast('cuda', dtype=torch.float16), torch.no_grad():
                video_frames = pipe(**kwargs)[0]

    except Exception as e:
        print(f"Inference error: {e}")
        print("Trying with fallback configuration...")
        
        # Configuration de fallback plus conservative
        kwargs["num_frames"] = min(kwargs["num_frames"], 16)
        kwargs["num_inference_steps"] = min(kwargs["num_inference_steps"], 8)
        
        # Disable mixed precision pour éviter les erreurs de dtype
        with torch.no_grad():
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