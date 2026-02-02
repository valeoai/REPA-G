# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Large-scale generation script for SiT with MultiPotential conditioning.
Efficiently batches samples for evaluation (e.g., 50k ImageNet).

Modes:
- multipot-repa: Anchor (Repa) + Target (Repa)
- multipot-transport: Anchor (Repa) + Target (Transport)
- multipot-free-energy: Anchor (Repa) + Target (Free Energy)
- baseline-uncond: Unconditional generation
- baseline-class: Class-conditional generation
- baseline-interp: Interpolation between Anchor Class and Target Class (Label mixing)
"""

import argparse
import gc
import json
import math
import os
import shutil
import yaml
from datetime import datetime
from pathlib import Path
import re

from dictdot import dictdot
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from tqdm import tqdm
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models.sit import SiT_models
from models.autoencoder import vae_models
from samplers import euler_sampler, euler_maruyama_sampler
from utils import load_encoders, denormalize_latents
from potential import TransportPotential, FreeEnergy, RepaPotential, MultiPotential
from dataset import CustomINH5PathDataset, CustomINH5ClassDataset
from train_repae import preprocess_raw_image

def get_balanced_imagenet_classes():
    """
    Returns 50 ImageNet classes optimized for background swapping.
    Focuses on "portable" objects that naturally sit on surfaces 
    to avoid scale/context ambiguity (e.g., no airplanes or boats).
    """
    balanced_classes = {
        # Distinct, ground-dwelling or perched animals
        "Animals": [
            207, # Golden retriever (Good texture/fur)
            281, # Tabby cat (Classic distinct object)
            340, # Zebra (High contrast pattern)
            30,  # Bullfrog (Sit-on-surface pose)
            76,  # Tarantula (Distinct silhouette)
            1,   # Goldfish (Often in bowls, but distinct orange color)
            323, # Monarch butterfly (Distinct pattern)
            143, # Crane (Bird) - Standing pose allows for ground texture
            22,  # Bald eagle
            292, # Tiger
        ],
        # Small vehicles & Movable machinery (Replaced Planes/Boats)
        "Movables": [
            444, # Bicycle (Open structure shows background well)
            670, # Motor scooter
            671, # Mountain bike
            429, # Baby stroller (Distinct structure)
            621, # Lawnmower (Sits on grass/ground naturally)
            799, # Shopping cart (Wireframe shows background)
            872, # Tricycle
            479, # Car wheel (Distinct shape)
            555, # Fire engine (Retained as distinct red object, usually on ground)
            817, # Sports car (Low to ground, good for "on road/brick")
        ],
        # Fruits & Small Foods (Excellent for texture contrast)
        "Food_Flora": [
            949, # Strawberry
            954, # Banana
            948, # Granny Smith (Apple)
            923, # Plate (Flat object, good for surface interactions)
            963, # Pizza
            933, # Cheeseburger
            992, # Agaric (Mushroom)
            988, # Acorn
            965, # Burrito
            309, # Bee (Small, distinct yellow/black)
        ],
        # Distinct Tools & Tech
        "Tools_Tech": [
            402, # Acoustic guitar
            464, # Broom (Long handle, minimal occlusion)
            587, # Hammer
            784, # Screwdriver
            805, # Soccer ball
            852, # Tennis racket
            518, # Crash helmet (Distinct round shape)
            892, # Wall clock (Round, distinct)
            620, # Laptop
            722, # Ping-pong ball
        ],
        # Household Containers & Wearables (Replaced large Tables)
        "Household": [
            849, # Teapot (Classic CV object)
            505, # Coffeepot
            859, # Toaster (Boxy, chrome reflections)
            883, # Vase
            515, # Cowboy hat
            769, # Running shoe (Great for "on grass/brick")
            414, # Backpack
            879, # Umbrella
            850, # Teddy bear
            417, # Balloon
        ]
    }
    
    # Flatten the list
    all_ids = []
    for category in balanced_classes.values():
        all_ids.extend(category)
        
    return ",".join(map(str, all_ids))

def extract_encoder_features(encoder, raw_image, encoder_type, return_cls=False):
    # (Same as before)
    z = encoder.forward_features(raw_image)
    if 'dinov2' in encoder_type:
        return z['x_norm_clstoken'] if return_cls else z['x_norm_patchtokens']
    elif 'mocov3' in encoder_type:
        return z[:, 0] if return_cls else z[:, 1:]
    return z

def load_representative_features_list(file_path):
    # (Same as before, simplified printing)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Representative features list not found: {file_path}")

    file_path = Path(file_path)
    features_list = file_path.glob("*.pt")
    pattern = r"^class_(?P<class_id>\d+)_sample_(?P<sample_id>\d+)_(?P<prompt>.+)\.pt$"
    representative_features = []

    for feature_file_name in features_list:
        basename = os.path.basename(feature_file_name)
        match = re.search(pattern, basename)
        if match:
            data = match.groupdict()
            # Defer loading tensor to main loop to save memory if list is huge
            # But for speed, we load here if memory allows.
            representative_features.append({
                "path": str(feature_file_name),
                "prompt": data['prompt'], 
                "class_id": int(data['class_id']), 
                "sample_id": int(data['sample_id'])
            })
    return representative_features

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    
    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    device = f"cuda:{device_id}"
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device_id)

    # Load Config
    if args.exp_path is None or args.train_steps is None:
        raise ValueError("Must provide exp_path and train_steps")

    with open(os.path.join(args.exp_path, "args.json"), "r") as f:
        config = dictdot(json.load(f))

    # Determine Latent Size
    if config.vae == "f8d4":
        latent_size = config.resolution // 8
        in_channels = 4
    elif config.vae == "f16d32":
        latent_size = config.resolution // 16
        in_channels = 32
    else:
        raise NotImplementedError()

    # Load Model
    encoders, _, _ = load_encoders(config.enc_type, "cpu", config.resolution)
    z_dims = [encoder.embed_dim for encoder in encoders] if config.enc_type != 'None' else [0]
    del encoders
    gc.collect()

    block_kwargs = {"fused_attn": config.fused_attn, "qk_norm": config.qk_norm}
    model = SiT_models[config.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=config.num_classes,
        class_dropout_prob=config.cfg_prob,
        z_dims=z_dims,
        encoder_depth=config.encoder_depth,
        bn_momentum=config.bn_momentum,
        **block_kwargs,
    ).to(device)

    train_step_str = str(args.train_steps).zfill(7)
    state_dict = torch.load(
        os.path.join(args.exp_path, "checkpoints", train_step_str +'.pt'),
        map_location=device,
    )
    model.load_state_dict(state_dict['ema'])
    model.eval()

    # Load VAE
    vae = vae_models[config.vae]().to(device)
    if "vae" in state_dict:
        vae_state_dict = state_dict['vae']
        latents_scale = state_dict["ema"]["bn.running_var"].rsqrt().view(1, in_channels, 1, 1).to(device)
        latents_bias = state_dict["ema"]["bn.running_mean"].view(1, in_channels, 1, 1).to(device)
    else:
        vae_state_dict = torch.load(config.vae_ckpt, map_location=device)
        latents_stats = torch.load(config.vae_ckpt.replace(".pt", "-latents-stats.pt"), map_location=device)
        latents_scale = latents_stats["latents_scale"].to(device)
        latents_bias = latents_stats["latents_bias"].to(device)
        del latents_stats

    vae.load_state_dict(vae_state_dict)
    vae.eval()
    
    del state_dict, vae_state_dict
    gc.collect()
    torch.cuda.empty_cache()

    # Determine if we need the Anchor Encoder (DINOv2)
    use_multipotential = "multipot" in args.method
    encoder = None
    if use_multipotential:
        if rank == 0:
            print(f"Loading DINOv2 encoder for Anchor conditioning...")
            encoders_list, _, _ = load_encoders("dinov2-vit-b", device, config.resolution)
            encoder = encoders_list[0]
            encoder.eval()
        dist.barrier()
        if rank != 0:
            encoders_list, _, _ = load_encoders("dinov2-vit-b", device, config.resolution)
            encoder = encoders_list[0]
            encoder.eval()

    # -------------------------------------------------------------------------
    # Dataset and Anchor Selection
    # -------------------------------------------------------------------------
    class_sorted_dataset = None
    selected_reference_samples = {}
    
    # 1. Parse Classes
    if args.classes_to_sample == "balanced":
        classes_str = get_balanced_imagenet_classes()
        classes_to_sample = [int(c.strip()) for c in classes_str.split(',') if c.strip()]
    elif args.classes_to_sample == "all":
        classes_to_sample = list(range(1000))
    else:
        classes_to_sample = [int(c.strip()) for c in args.classes_to_sample.split(',') if c.strip()]

    use_multipotential = "multipot" in args.method

    # 2. Setup Dataset & Anchors (Metadata Only)
    if rank == 0: print("Initializing Dataset (Metadata)...")
    # This is now fast because it only reads JSON metadata
    class_sorted_dataset = CustomINH5ClassDataset(args.data_dir)

    # Load pre-computed anchor indices (JSON)
    if not args.anchor_indices_path or not os.path.exists(args.anchor_indices_path):
        raise FileNotFoundError(f"Anchor indices file not found at: {args.anchor_indices_path}")

    if rank == 0: print(f"Loading anchor map from {args.anchor_indices_path}...")
    with open(args.anchor_indices_path, 'r') as f:
        loaded_anchors = json.load(f)

    # Filter loaded anchors to only the classes we want
    for cid in classes_to_sample:
        str_cid = str(cid)
        if str_cid in loaded_anchors:
            selected_reference_samples[cid] = loaded_anchors[str_cid]

    # -------------------------------------------------------------------------
    # Task Scheduling (Lazy)
    # -------------------------------------------------------------------------
    all_tasks = []
    print("Creating Task List...")

    rep_features_info = load_representative_features_list(args.representative_features_list)
    
    # Strategy: Iterate Anchors -> Rep Features.
    # This ensures tasks using the same Anchor Image are generated sequentially.
    for class_id in classes_to_sample:
        if class_id not in selected_reference_samples: continue
        
        # The JSON contains local indices relative to the class (0 to ~1300)
        local_indices = selected_reference_samples[class_id]
        
        for local_idx in local_indices:
            # Resolve Global Index immediately (Int lookup, very fast)
            try:
                global_idx = class_sorted_dataset.get_global_index(class_id, local_idx)
            except ValueError:
                continue # Skip invalid indices
            
            # Create a task for every Representative Feature
            for rep_info in rep_features_info:
                all_tasks.append({
                    'global_idx': global_idx, # Crucial for caching
                    'anchor_class_id': class_id,
                    'anchor_local_idx': local_idx,
                    'rep_feat_info': rep_info,
                })

    # Shard Tasks
    num_tasks = len(all_tasks)
    my_task_indices = list(range(rank, num_tasks, dist.get_world_size()))
    my_tasks = [all_tasks[i] for i in my_task_indices]

    if rank == 0:
        print(f"Total Tasks: {num_tasks}. GPU {rank} has {len(my_tasks)} tasks.")

    sample_folder_dir = (f"{args.sample_dir}/{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_steps_{args.num_steps}_al_{args.anchor_lambda}_pl_{args.param_lambda}_po_{args.param_other}_{args.mode}")
    images_dir = os.path.join(sample_folder_dir, "images")
    if rank == 0:
        os.makedirs(images_dir, exist_ok=True)
        # Save config
        config_dict = {'args': vars(args), 'model_config': dict(config)}
        with open(os.path.join(sample_folder_dir, "config.yaml"), 'w') as f:
            yaml.dump(config_dict, f)
    dist.barrier()
    # -------------------------------------------------------------------------
    # Main Generation Loop with Caching
    # -------------------------------------------------------------------------
    # ... (Output directory setup) ...
    
    batch_size = args.batch_size
    num_batches = math.ceil(len(my_tasks) / batch_size)

    # LRU-style cache for Anchor Features
    # Since tasks are sorted by Anchor, a size of 1 is often sufficient.
    # Key: global_idx -> Value: tensor (1, N, D)
    anchor_cache = {}
    MAX_CACHE = 100

    for b in tqdm(range(num_batches), desc=f"Rank {rank} Batches"):
        batch_tasks = my_tasks[b*batch_size : (b+1)*batch_size]
        curr_batch_size = len(batch_tasks)
        if curr_batch_size == 0: break

        # 1. Prepare Inputs
        z = torch.randn(curr_batch_size, model.in_channels, latent_size, latent_size, device=device)
        y_labels = torch.zeros(curr_batch_size, dtype=torch.long, device=device)
        
        potential = None
        
        # Arrays to store metadata for saving
        meta_data = []
        y_bis = None

        if use_multipotential:
            z.requires_grad_(True)
            anchor_feats_list = []
            target_feats_list = []
            
            for i, task in enumerate(batch_tasks):
                # 1. Labels
                y_labels[i] = task['anchor_class_id'] if args.use_class_label else 1000

                # 2. Get Anchor Feature (Lazy Load + Cache)
                global_idx = task['global_idx']
                
                if global_idx in anchor_cache:
                    # CACHE HIT
                    a_feat = anchor_cache[global_idx]
                else:
                    # CACHE MISS - Load from Disk
                    # This is the only time we touch the H5 file
                    raw_img_tensor, _ = class_sorted_dataset[global_idx]
                    raw_img_tensor = raw_img_tensor.unsqueeze(0).to(device)
                    raw_img_pre = preprocess_raw_image(raw_img_tensor, config.enc_type)
                    
                    with torch.no_grad():
                        a_feat = extract_encoder_features(encoder, raw_img_pre, config.enc_type)
                        a_feat = torch.nn.functional.normalize(a_feat, dim=-1)
                    
                    # Update Cache
                    if len(anchor_cache) >= MAX_CACHE:
                        anchor_cache.pop(next(iter(anchor_cache)))
                    anchor_cache[global_idx] = a_feat

                anchor_feats_list.append(a_feat)

                # 3. Load Target Feature
                # (Load from small .pt files is fast)
                t_feat = torch.load(task['rep_feat_info']['path'], map_location=device)
                if len(t_feat.shape) == 1: t_feat = t_feat.unsqueeze(0)
                t_feat = torch.nn.functional.normalize(t_feat, dim=-1)
                target_feats_list.append(t_feat)
                
                meta_data.append(task)
            
            # ... (Stack features, create Potentials, Run Sampling) ...
            # Same logic as previous script, just using anchor_feats_list populated above
            anchor_cond_batch = torch.cat(anchor_feats_list, dim=0)
            target_cond_batch = torch.cat(target_feats_list, dim=0).unsqueeze(1)

            anchor_potential = RepaPotential(
                cond=anchor_cond_batch, 
                lamda=torch.full((curr_batch_size,), args.anchor_lambda, device=device)
            )
            
            target_lam_tensor = torch.full((curr_batch_size,), args.param_lambda, device=device)
            target_other_tensor = torch.full((curr_batch_size,), args.param_other, device=device)

            if "transport" in args.method:
                target_potential = TransportPotential(cond=target_cond_batch, lamda=target_lam_tensor, eps=target_other_tensor)
            elif "free-energy" in args.method:
                target_potential = FreeEnergy(cond=target_cond_batch, lamda=target_lam_tensor, T=target_other_tensor)
            elif "repa" in args.method:
                target_potential = RepaPotential(cond=target_cond_batch, lamda=target_lam_tensor)
            
            potential = MultiPotential([anchor_potential, target_potential], secondary_potential_guidance_threshold=args.secondary_potential_guidance_threshold)
            
        else:
            if args.method == "baseline-interp":
                y_bis = torch.zeros(curr_batch_size, dtype=torch.long, device=device)
            # Baselines
            for i, task in enumerate(batch_tasks):
                cid = task['anchor_class_id']
                rep_feat_info = task['rep_feat_info']
                if args.method == "baseline-uncond":
                    y_labels[i] = 1000
                elif args.method == "baseline-class":
                    y_labels[i] = cid
                elif args.method == "baseline-interp":
                    y_labels[i] = cid
                    y_bis[i] = rep_feat_info['class_id']
                    
                meta_data.append(task)

        # 2. Sampling
        sampling_kwargs = dict(
            model=model, 
            latents=z,
            y=y_labels,
            y_bis=y_bis,
            potential=potential, 
            num_steps=args.num_steps, 
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
            gibbs=True,
        )

        if args.mode == "sde":
            samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
        else:
            samples = euler_sampler(**sampling_kwargs).to(torch.float32)

        # 3. Decode and Save
        with torch.no_grad():
            samples = vae.decode(denormalize_latents(samples, latents_scale, latents_bias)).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(255. * samples, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for j, sample_img in enumerate(samples):
                task = meta_data[j]
                
                fname = (f"method_{args.method}_ac_{task['anchor_class_id']}_as_{task['anchor_local_idx']}_"
                            f"tc_{task['rep_feat_info']['class_id']}_ts_{task['rep_feat_info']['sample_id']}_"
                            f"prompt_{task['rep_feat_info']['prompt']}.png")
                
                Image.fromarray(sample_img).save(os.path.join(images_dir, fname))

    print(f"Rank {rank} finished.")
    dist.barrier()
    if rank == 0:
        print("All sampling finished.")
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Method Selection
    parser.add_argument("--method", type=str, required=True,
                        choices=["multipot-repa", "multipot-transport", "multipot-free-energy", 
                                 "baseline-uncond", "baseline-class", "baseline-interp"],
                        help="Generation method to use.")

    # Fixed Hyperparameters (No longer grid search)
    parser.add_argument("--param-lambda", type=float, default=1.0, help="Lambda for Target Potential")
    parser.add_argument("--param-other", type=float, default=1.0, help="Epsilon/Temp for Target Potential")
    parser.add_argument("--anchor-lambda", type=float, default=1.0, help="Lambda for Anchor Potential")
    parser.add_argument("--secondary-potential-guidance-threshold", type=float, default=0.0)

    # Configuration
    parser.add_argument("--representative-features-list", type=str, default=None,
                        help="Path to .txt file containing list of .npz files (Required for multipot methods).")
    parser.add_argument("--classes-to-sample", type=str, default="balanced", help="'balanced', 'all', or comma-separated list")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples per class/pair")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--anchor-indices-path", type=str, default=None,
                    help="Path to the JSON file containing pre-selected anchor indices.")
    # Standard Arguments
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--exp-path", type=str, required=True)
    parser.add_argument("--train-steps", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--anchor-seed", type=int, default=42)
    
    # Sampler Args
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--path-type", type=str, default="linear")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)
    parser.add_argument("--use-class-label", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    main(args)