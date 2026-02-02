# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

Ref:
    https://github.com/sihyun-yu/REPA/blob/main/generate.py
"""

import argparse
import gc
import json
import math
import os
import shutil
import yaml
import fcntl
from datetime import datetime, timedelta

from dictdot import dictdot
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from tqdm import tqdm
from torchvision.transforms import Normalize, transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Install with: pip install mlflow")

try:
    import tensorflow.compat.v1 as tf
    from evaluator import Evaluator
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")

from models.sit import SiT_models
from models.autoencoder import vae_models
from samplers import euler_sampler, euler_maruyama_sampler
from utils import load_encoders, denormalize_latents, load_sit_and_vae, extract_sit_features
from potential import RepaPotential, MeanFeatAlignment
from dataset import CustomINH5Dataset
from dataset_coco import SimpleCOCODataset, COCONpzDataset
from train_repae import preprocess_raw_image


def extract_encoder_features(encoder, raw_image, encoder_type, average_features=False):
    """Extract features from encoder with proper postprocessing.
    
    Args:
        encoder: Vision encoder model
        raw_image: Preprocessed image tensor
        encoder_type: Type of encoder (e.g., 'dinov2', 'mocov3', 'clip')
        average_features: Whether to average features spatially (default: False)
    
    Returns:
        Extracted features tensor [B, num_patches, D] or [B, 1, D] if averaged
    """
    z = encoder.forward_features(raw_image)
    if 'mocov3' in encoder_type:
        z = z[:, 1:]  # Remove CLS token
    elif 'dinov2' in encoder_type:
        z = z['x_norm_patchtokens']
    
    if average_features:
        z = z.mean(dim=1, keepdim=True)  # Average spatially: [B, N, D] -> [B, 1, D]
    
    return z


def compute_psnr(img1, img2, max_val=255.0):
    """Compute PSNR between two images.
    
    Args:
        img1: First image tensor (B, H, W, C) or (B, C, H, W) in range [0, 255]
        img2: Second image tensor with same shape as img1
        max_val: Maximum pixel value (default: 255.0)
    
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1.float() - img2.float()) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def compute_psnr_with_mask(img1, img2, mask, max_val=255.0):
    """Compute PSNR between two images, considering only masked regions.
    
    Args:
        img1: First image tensor (B, H, W, C) or (B, C, H, W) in range [0, 255]
        img2: Second image tensor with same shape as img1
        mask: Binary mask tensor (B, L) where L is number of patches. 1 for masked, 0 for unmasked.
              Will be reshaped to (B, H, W) for pixel-wise masking.
        max_val: Maximum pixel value (default: 255.0)
    
    Returns:
        PSNR value in dB
    """
    B, C, H, W = img1.shape
    
    # Reshape mask from (B, L) to (B, H, W)
    mask_2d = mask.reshape(B, int(np.sqrt(mask.shape[1])), int(np.sqrt(mask.shape[1])))
    # Resize mask to image resolution (H, W) using nearest neighbor
    mask_resized = torch.nn.functional.interpolate(
        mask_2d.unsqueeze(1).float(), size=(H, W), mode='nearest'
    ).squeeze(1) # Shape (B, H, W)

    # Expand mask to cover all color channels
    mask_expanded = mask_resized.unsqueeze(1).expand_as(img1) # Shape (B, C, H, W)

    # Apply mask
    masked_img1 = img1.float() * mask_expanded
    masked_img2 = img2.float() * mask_expanded

    # Compute MSE only on masked pixels
    diff = masked_img1 - masked_img2
    squared_diff = diff ** 2
    mse = torch.mean(squared_diff[mask_expanded > 0])
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def compute_batch_psnr_with_mask(imgs1, imgs2, masks, max_val=255.0):
    """Compute PSNR for a batch of images with masks, returning individual PSNR values.
    
    Args:
        imgs1: First image batch (B, C, H, W) in range [0, 255]
        imgs2: Second image batch (B, C, H, W) in range [0, 255]
        masks: Binary mask tensor (B, L) where L is number of patches
        max_val: Maximum pixel value (default: 255.0)
    
    Returns:
        List of PSNR values in dB, one per image in batch
    """
    B, C, H, W = imgs1.shape
    
    # Reshape and resize all masks at once (much faster than per-image)
    mask_2d = masks.reshape(B, int(np.sqrt(masks.shape[1])), int(np.sqrt(masks.shape[1])))
    mask_resized = torch.nn.functional.interpolate(
        mask_2d.unsqueeze(1).float(), size=(H, W), mode='nearest'
    ).squeeze(1)  # Shape (B, H, W)
    
    # Expand mask to cover all color channels
    mask_expanded = mask_resized.unsqueeze(1).expand_as(imgs1)  # Shape (B, C, H, W)
    
    # Apply mask to all images at once
    masked_imgs1 = imgs1.float() * mask_expanded
    masked_imgs2 = imgs2.float() * mask_expanded
    
    # Compute squared differences for all images
    diff = masked_imgs1 - masked_imgs2
    squared_diff = diff ** 2  # Shape (B, C, H, W)
    
    # Compute MSE per image (sum over C, H, W dimensions, then divide by number of masked pixels)
    # Reshape to (B, -1) to sum over all spatial and channel dimensions per image
    squared_diff_flat = squared_diff.reshape(B, -1)  # (B, C*H*W)
    mask_expanded_flat = (mask_expanded > 0).reshape(B, -1)  # (B, C*H*W)
    
    # Sum squared errors and count masked pixels per image
    sum_squared_errors = (squared_diff_flat * mask_expanded_flat.float()).sum(dim=1)  # (B,)
    num_masked_pixels = mask_expanded_flat.sum(dim=1).float()  # (B,)
    
    # Compute MSE per image
    mse_per_image = sum_squared_errors / num_masked_pixels.clamp(min=1)  # (B,)
    
    # Compute PSNR per image
    psnr_per_image = 20 * torch.log10(max_val / torch.sqrt(mse_per_image.clamp(min=1e-10)))
    
    return psnr_per_image.cpu().tolist()

def register_experiment(output_dir, config_dict, registry_path="experiments_registry.json"):
    """
    Register a generation experiment in a central JSON file with file locking for concurrent jobs.
    
    Args:
        output_dir: Path to the output directory
        config_dict: Configuration dictionary with experiment parameters
        registry_path: Path to the registry JSON file
    """
    registry_entry = {
        'output_dir': os.path.abspath(output_dir),
        'timestamp': config_dict['timestamp'],
        'experiment_name': config_dict['experiment_name'],
        'model': config_dict['training_config'].get('model', 'unknown'),
        'train_steps': config_dict['generation_args'].get('train_steps', 'unknown'),
        'num_samples': config_dict['generation_args'].get('num_fid_samples', 0),
        'cfg_scale': config_dict['generation_args'].get('cfg_scale', 1.0),
        'use_feature_conditioning': config_dict['generation_args'].get('use_feature_conditioning', False),
        'repa_lambda': config_dict['generation_args'].get('repa_lambda', None),
    }

    # Use file locking to safely handle concurrent writes
    lock_path = registry_path + '.lock'
    with open(lock_path, 'w') as lock_file:
        # Acquire exclusive lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        
        try:
            # Load existing registry or create new one
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    try:
                        registry = json.load(f)
                        if not isinstance(registry, list):
                            registry = []
                    except json.JSONDecodeError:
                        registry = []
            else:
                registry = []
            
            # Append new entry
            registry.append(registry_entry)
            
            # Write back the complete registry atomically
            temp_path = registry_path + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(registry, f, indent=2)
            os.replace(temp_path, registry_path)
            
            print(f"Registered experiment in {registry_path}")
        finally:
            # Release lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    
    # Clean up lock file if possible
    try:
        os.remove(lock_path)
    except:
        pass


def create_npz_from_sample_folder(sample_dir, images_subdir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    
    Args:
        sample_dir: Root output directory
        images_subdir: Subdirectory containing images
        num: Number of samples to include
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{images_subdir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = os.path.join(sample_dir, "img.npz")
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    # Don't disable gradients globally if using feature conditioning
    if not args.use_feature_conditioning:
        torch.set_grad_enabled(False)

    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    device = f"cuda:{device_id}"
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device_id)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Initialize MLflow logging (only on rank 0)
    mlflow_run_started = False
    if rank == 0 and MLFLOW_AVAILABLE and args.use_mlflow:
        # Set MLflow tracking URI if specified
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
            print(f"MLflow tracking URI set to: {args.mlflow_tracking_uri}")
        
        mlflow.set_experiment(args.mlflow_experiment_name)
        
        # Start MLflow run with a temporary name (will be updated later with proper folder name)
        if args.mlflow_run_name:
            mlflow.start_run(run_name=args.mlflow_run_name)
        else:
            mlflow.start_run()  # Start with auto-generated name, will rename later
        mlflow_run_started = True
        print(f"MLflow run started")
    
    if args.train_steps is not None:
        with open(os.path.join(args.exp_path, "args.json"), "r") as f:
            config = dictdot(json.load(f))
    else:
        # For single checkpoint file, config parsing is handled internally
        config = None

    # Load SiT model and VAE (config parsing handled internally)
    exp_name = os.path.basename(args.exp_path)
    # Remove extension if it exists (handles both .pt and .pth)
    if args.exp_path.endswith('.pt') or args.exp_path.endswith('.pth'):
        exp_name = os.path.splitext(exp_name)[0]
    if args.train_steps is None:
        checkpoint_path = args.exp_path
        train_step_str = None
    else:
        train_step_str = str(args.train_steps).zfill(7)
        checkpoint_path = os.path.join(args.exp_path, "checkpoints", train_step_str +'.pt')
    
    model, vae, latents_scale, latents_bias, _, latent_size, config = load_sit_and_vae(
        checkpoint_path=checkpoint_path,
        device=device,
        config=config,
    )
    
    # Log parameters to MLflow
    if rank == 0 and MLFLOW_AVAILABLE and args.use_mlflow and mlflow_run_started:
        try:
            # Log generation arguments
            mlflow.log_params({
                "exp_path": args.exp_path,
                "train_steps": args.train_steps,
                "global_seed": args.global_seed,
                "num_fid_samples": args.num_fid_samples,
                "pproc_batch_size": args.pproc_batch_size,
                "mode": args.mode,
                "num_steps": args.num_steps,
                "cfg_scale": args.cfg_scale,
                "guidance_low": args.guidance_low,
                "guidance_high": args.guidance_high,
                "path_type": args.path_type,
                "heun": args.heun,
                "gibbs": args.gibbs,
                "label_sampling": args.label_sampling,
                "use_feature_conditioning": args.use_feature_conditioning,
                "use_uncond_class": args.use_uncond_class,
            })
            print("MLflow: Logged generation parameters")
            
            # Log feature conditioning parameters if enabled
            if args.use_feature_conditioning:
                mlflow.log_params({
                    "feature_type": args.feature_type,
                    "repa_lambda": args.repa_lambda,
                    "anchor_seed": args.anchor_seed,
                    "average_features": args.average_features,
                    "use_pca_feature_mask": args.use_pca_feature_mask,
                    "compute_similarity": args.compute_similarity,
                    "compute_conditioning_alignment": args.compute_conditioning_alignment,
                })
                if args.feature_type == 'sit':
                    mlflow.log_params({
                        "sit_depth": args.sit_depth,
                        "use_projector": args.use_projector,
                    })
                if args.additional_similarity_backbones:
                    mlflow.log_param("additional_similarity_backbones", args.additional_similarity_backbones)
                print("MLflow: Logged feature conditioning parameters")
            
            # Log model config
            mlflow.log_params({
                "model": config.model,
                "vae": config.vae,
                "resolution": config.resolution,
                "num_classes": config.num_classes,
            })
            print("MLflow: Logged model config")
        except Exception as e:
            print(f"ERROR: Failed to log parameters to MLflow: {e}")
            import traceback
            traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()

    assert args.cfg_scale >= 1.0, "cfg_scale should be >= 1.0"

    # Setup feature conditioning if enabled
    encoder = None
    additional_encoders = {}  # Dict to store additional encoders for similarity computation
    dataset = None
    anchor_indices = None
    anchor_labels = None
    anchor_images_dict = {}  # Store anchor images for PSNR computation
    conditioning_alignment_list = []  # Store alignment with conditioning features and PSNR values
    backbone_similarities_lists = {}  # Store similarities for all similarity backbones
    use_sdvae = isinstance(vae, type(vae)) and hasattr(vae, 'config')  # Detect SD-VAE
    pca_masks_data = None  # Store precomputed PCA masks
    
    if args.use_feature_conditioning:
        # Load encoder based on feature type
        if args.feature_type == 'sit':
            # For SiT features, we use the model itself, no separate encoder needed
            if rank == 0:
                print(f"Using SiT features (depth={args.sit_depth}, projector={'enabled' if args.use_projector else 'disabled'}) for conditioning...")
            encoder = None  # Will use model directly
        else:
            # Load DINOv2 encoder (rank 0 loads first, then others)
            if rank == 0:
                print(f"Loading DINOv2 encoder for feature conditioning...")
                encoders_list, _, _ = load_encoders("dinov2-vit-b", device, config.resolution)
                encoder = encoders_list[0]
                encoder.eval()
            
            # Wait for rank 0 to finish downloading/loading the model
            dist.barrier()
            
            # Now other ranks can load safely
            if rank != 0:
                encoders_list, _, _ = load_encoders("dinov2-vit-b", device, config.resolution)
                encoder = encoders_list[0]
                encoder.eval()
        
        # Load encoders for similarity computation if specified
        if args.additional_similarity_backbones:
            backbone_names = [b.strip() for b in args.additional_similarity_backbones.split(',') if b.strip()]
            for backbone_name in backbone_names:
                if rank == 0:
                    print(f"Loading {backbone_name} encoder for similarity computation...")
                    try:
                        additional_enc_list, _, _ = load_encoders(backbone_name, device, config.resolution)
                        additional_encoders[backbone_name] = additional_enc_list[0]
                        additional_encoders[backbone_name].eval()
                        backbone_similarities_lists[backbone_name] = []
                    except Exception as e:
                        print(f"Warning: Failed to load encoder {backbone_name}: {e}")
                
                dist.barrier()
                
                if rank != 0 and backbone_name not in additional_encoders:
                    try:
                        additional_enc_list, _, _ = load_encoders(backbone_name, device, config.resolution)
                        additional_encoders[backbone_name] = additional_enc_list[0]
                        additional_encoders[backbone_name].eval()
                        backbone_similarities_lists[backbone_name] = []
                    except Exception as e:
                        print(f"Warning: Failed to load encoder {backbone_name}: {e}")
        
        # Load dataset based on type
        if args.dataset_type == "imagenet":
            if rank == 0:
                print("Loading ImageNet dataset for anchor images...")
            dataset = CustomINH5Dataset(args.data_dir)
            dataset_suffix = "imagenet"
        elif args.dataset_type == "coco":
            if rank == 0:
                print(f"Loading COCO {args.coco_split}{args.coco_year} dataset for anchor images...")
            dataset = SimpleCOCODataset(
                root=args.data_dir,
                split=args.coco_split,
                year=args.coco_year,
                image_size=config.resolution,
                transform= transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.PILToTensor(),
                ]),
            )
            dataset_suffix = f"coco_{args.coco_split}{args.coco_year}"
        elif args.dataset_type == "coco-npz":
            if rank == 0:
                print(f"Loading COCO NPZ dataset for anchor images...")
            dataset = COCONpzDataset(args.data_dir)
            dataset_suffix = "coco-npz"
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset_type}")
        
        # Load or create anchor indices and labels with dataset-specific naming
        if args.dataset_type == "imagenet":
            anchor_labels_path = os.path.join(
            args.sample_dir,
            f"anchor_labels_{args.num_fid_samples}_seed{args.anchor_seed}.npy"
            )
            anchor_indices_path = os.path.join(
            args.sample_dir,
            f"anchor_indices_{args.num_fid_samples}_seed{args.anchor_seed}.npy"
            )
        else:
            anchor_labels_path = os.path.join(
            args.sample_dir,
            f"anchor_labels_{dataset_suffix}_{args.num_fid_samples}_seed{args.anchor_seed}.npy"
            )
            anchor_indices_path = os.path.join(
            args.sample_dir,
            f"anchor_indices_{dataset_suffix}_{args.num_fid_samples}_seed{args.anchor_seed}.npy"
            )

        os.makedirs(args.sample_dir, exist_ok=True)
        
        if rank == 0:
            if os.path.exists(anchor_indices_path) and os.path.exists(anchor_labels_path):
                anchor_indices = np.load(anchor_indices_path)
                anchor_labels = np.load(anchor_labels_path)
                print(f"Loaded existing anchor indices and labels from {anchor_indices_path}")
            else:
                # Create anchor indices and labels that match the label sampling strategy
                # We need to first establish the label distribution
                if args.label_sampling == "equal":
                    # Equal distribution: each class appears exactly per_class times
                    per_class = args.num_fid_samples // args.num_classes
                    desired_labels = torch.arange(args.num_classes).repeat_interleave(per_class)
                    gen = torch.Generator().manual_seed(args.global_seed)
                    desired_labels = desired_labels[torch.randperm(desired_labels.numel(), generator=gen)]
                elif args.label_sampling == "random":
                    # Random distribution
                    desired_labels = torch.randint(0, args.num_classes, (args.num_fid_samples,))
                else:
                    raise NotImplementedError(f"Unknown label_sampling: {args.label_sampling}")
                
                # For each desired label, find a random image from dataset with that label
                anchor_indices_list = []
                anchor_labels_list = []
                rng = np.random.RandomState(args.anchor_seed)
                
                if args.dataset_type == "imagenet":
                    # Build a mapping of label -> list of dataset indices efficiently using dataset.labels
                    print("Building label-to-index mapping from dataset labels...")
                    dataset_labels = dataset.labels  # Already loaded as numpy array
                    label_to_indices = {}
                    for label in range(args.num_classes):
                        label_to_indices[label] = np.where(dataset_labels == label)[0]
                    
                    # Select anchors according to desired labels
                    for desired_label in desired_labels:
                        label = int(desired_label)
                        if label in label_to_indices and len(label_to_indices[label]) > 0:
                            # Randomly select an image with this label
                            selected_idx = rng.choice(label_to_indices[label])
                            anchor_indices_list.append(selected_idx)
                            anchor_labels_list.append(label)
                        else:
                            # Fallback: if no image with this label exists, use a random image
                            selected_idx = rng.choice(len(dataset))
                            actual_label = dataset_labels[selected_idx]
                            anchor_indices_list.append(selected_idx)
                            anchor_labels_list.append(int(actual_label))
                elif args.dataset_type == "coco":
                    # For COCO, just randomly sample images (no class labels available)
                    print("Randomly selecting anchor images from COCO dataset...")
                    num_available = len(dataset)
                    if args.num_fid_samples > num_available:
                        print(f"Warning: Requested {args.num_fid_samples} samples but COCO only has {num_available} images.")
                        print(f"Will sample with replacement.")
                        selected_indices = rng.choice(num_available, size=args.num_fid_samples, replace=True)
                    else:
                        selected_indices = rng.choice(num_available, size=args.num_fid_samples, replace=False)
                    
                    anchor_indices_list = selected_indices.tolist()
                    # For COCO, use dummy labels (all zeros or all unconditional class)
                    anchor_labels_list = [1000] * args.num_fid_samples  # Unconditional class
                
                elif args.dataset_type == "coco-npz":
                    # Use all images in order from the NPZ dataset with no class labels
                    print("Selecting anchor images from COCO NPZ dataset...")
                    anchor_indices_list = list(range(len(dataset)))
                    anchor_labels_list = [1000] * len(dataset)  # Unconditional class
                
                anchor_indices = np.array(anchor_indices_list, dtype=np.int64)
                anchor_labels = np.array(anchor_labels_list, dtype=np.int64)
                
                np.save(anchor_indices_path, anchor_indices)
                np.save(anchor_labels_path, anchor_labels)
                print(f"Created and saved new anchor indices to {anchor_indices_path}")
                print(f"Created and saved new anchor labels to {anchor_labels_path}")
        else:
            anchor_indices = np.empty(args.num_fid_samples, dtype=np.int64)
            anchor_labels = np.empty(args.num_fid_samples, dtype=np.int64)
        
        # Broadcast anchor indices and labels to all ranks
        anchor_indices_tensor = torch.from_numpy(anchor_indices).to(device)
        anchor_labels_tensor = torch.from_numpy(anchor_labels).to(device)
        dist.broadcast(anchor_indices_tensor, src=0)
        dist.broadcast(anchor_labels_tensor, src=0)
        anchor_indices = anchor_indices_tensor.cpu().numpy()
        anchor_labels = anchor_labels_tensor.cpu().numpy()
        
        # Load precomputed PCA masks if requested
        if args.use_pca_feature_mask:
            # Derive mask file path from anchor file path
            # anchor_indices_imagenet_50000_seed42.npy -> pca_masks_imagenet_50000_seed42.npz
            anchor_basename = os.path.basename(anchor_indices_path)
            mask_basename = anchor_basename.replace("anchor_indices_", "pca_masks_").replace(".npy", ".npz")
            pca_masks_path = os.path.join(os.path.dirname(anchor_indices_path), mask_basename)
            
            # Check existence on rank 0, then all ranks load
            if rank == 0:
                if not os.path.exists(pca_masks_path):
                    raise FileNotFoundError(
                        f"PCA masks file not found: {pca_masks_path}\n"
                        f"Expected mask file matching anchor file: {anchor_indices_path}\n"
                        f"Please run: torchrun --nproc_per_node=N preprocess_pca_masks.py --anchor-file {anchor_indices_path}"
                    )
                print(f"Loading precomputed PCA masks from {pca_masks_path}")
            
            # Wait for rank 0 to check file existence
            dist.barrier()
            
            # All ranks load the mask file
            pca_masks_data = np.load(pca_masks_path)
            if rank == 0:
                print(f"Loaded {len(pca_masks_data.files)} precomputed masks")
        
        if rank == 0:
            print(f"Will compute features on-the-fly for {len(anchor_indices)} anchor images")
            print(f"Anchor labels distribution: min={anchor_labels.min()}, max={anchor_labels.max()}")

    # Create sample folder name using simplified naming scheme
    run_name_parts = [exp_name]
    if args.time == 1.0:
        run_name_parts.append("time1")
    if args.use_feature_conditioning:
        run_name_parts.append(f"lambda{args.repa_lambda}")
        run_name_parts.append(args.feature_type)
        if args.average_features:
            run_name_parts.append("average")
        if args.feature_type == 'sit' and args.use_projector:
            run_name_parts.append("proj")
        if args.use_pca_feature_mask:
            run_name_parts.append("mask")
    if args.use_uncond_class:
        run_name_parts.append("uncond")
    
    folder_name = "_".join(run_name_parts)
    sample_folder_dir = os.path.join(args.sample_dir, folder_name)
    
    # Update MLflow run name if it was auto-generated
    if rank == 0 and MLFLOW_AVAILABLE and args.use_mlflow and mlflow_run_started:
        if args.mlflow_run_name is None:
            # Set the run name to match the folder name
            mlflow.set_tag("mlflow.runName", folder_name)
            args.mlflow_run_name = folder_name
            print(f"MLflow run name set to: {args.mlflow_run_name}")
    
    # Create images subdirectory
    images_dir = os.path.join(sample_folder_dir, "images")
    
    skip = torch.tensor([False], device=device)
    if rank == 0:
        npz_path = os.path.join(sample_folder_dir, "img.npz")
        if os.path.exists(npz_path):
            skip[0] = True
            print(f"Skipping sampling as {npz_path} already exists.")
        else:
            os.makedirs(images_dir, exist_ok=True)
            print(f"Saving .png samples at {images_dir}")
            
            # Save experiment configuration as YAML
            config_dict = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'experiment_name': os.path.basename(sample_folder_dir),
                'generation_args': vars(args),
                'training_config': dict(config),
            }
            
            config_yaml_path = os.path.join(sample_folder_dir, "config.yaml")
            with open(config_yaml_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            print(f"Saved experiment configuration to {config_yaml_path}")
            
            # Register experiment in central registry
            registry_path = os.path.join(args.sample_dir, "experiments_registry.json")
            register_experiment(sample_folder_dir, config_dict, registry_path)
            
            # Copy anchor files to output directory if feature conditioning is used
            if args.use_feature_conditioning:
                # Determine dataset suffix for anchor file names
                if args.dataset_type == "imagenet":
                    dataset_suffix = "imagenet"
                elif args.dataset_type == "coco":
                    dataset_suffix = f"coco_{args.coco_split}{args.coco_year}"
                
                # Copy anchor indices
                dest_anchor_indices_path = os.path.join(
                    sample_folder_dir,
                    f"anchor_indices_{dataset_suffix}_{args.num_fid_samples}_seed{args.anchor_seed}.npy"
                )
                shutil.copy2(anchor_indices_path, dest_anchor_indices_path)
                print(f"Copied anchor indices to {dest_anchor_indices_path}")
                
                # Copy anchor labels
                dest_anchor_labels_path = os.path.join(
                    sample_folder_dir,
                    f"anchor_labels_{dataset_suffix}_{args.num_fid_samples}_seed{args.anchor_seed}.npy"
                )
                shutil.copy2(anchor_labels_path, dest_anchor_labels_path)
                print(f"Copied anchor labels to {dest_anchor_labels_path}")

    # Broadcast the skip flag to all processes
    dist.broadcast(skip, src=0)
    if skip.item():
        dist.destroy_process_group()
        return
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.pproc_batch_size
    world_size = dist.get_world_size()

    # Exact class balance: 50_000 images, 1_000 classes => 50 per class
    # Skip class balance check for COCO dataset
    if args.dataset_type == "imagenet" and not args.use_feature_conditioning:
        assert args.num_fid_samples % args.num_classes == 0, \
            f"num_fid_samples ({args.num_fid_samples}) must be divisible by num_classes ({args.num_classes})."
    per_class = args.num_fid_samples // args.num_classes  # 50 when 50k/1k

    # Build a global label schedule with exact counts, then (optionally) shuffle it.
    # IMPORTANT: all ranks must see the same permutation => use a rank-independent seed or broadcast.
    if rank == 0:
        if args.label_sampling == "equal" and args.dataset_type == "imagenet":
            y_all = torch.arange(args.num_classes, device=device).repeat_interleave(per_class)  # [0..999] each repeated 50x
            gen = torch.Generator(device=device).manual_seed(args.global_seed)  # SAME seed across ranks
            y_all = y_all[torch.randperm(y_all.numel(), generator=gen, device=device)]
        elif args.label_sampling == "random" or args.dataset_type == "coco" or args.dataset_type == "coco-npz":
            # For COCO or random sampling, generate random labels
            y_all = torch.randint(0, args.num_classes, (args.num_fid_samples,), device=device)
        else:
            raise NotImplementedError(f"Unknown label_sampling: {args.label_sampling}")
    else:
        y_all = torch.empty(args.num_fid_samples, device=device, dtype=torch.long)

    # Broadcast the global label schedule to all ranks
    dist.broadcast(y_all, src=0)

    # Equal shard per rank
    labels_per_rank = args.num_fid_samples // world_size   # 12_500 (4 GPUs) or 6_250 (8 GPUs)
    assert args.num_fid_samples % world_size == 0, \
        f"num_fid_samples ({args.num_fid_samples}) must be divisible by world_size ({world_size})."
    start = rank * labels_per_rank
    end = start + labels_per_rank
    y_this_rank = y_all[start:end]                         # shape: (labels_per_rank,)

    # Iteration planning with possible partial last batch (no need to force divisibility by n)
    total_to_make = y_this_rank.numel()                    # exactly 12,500 or 6,250
    iterations = int(math.ceil(total_to_make / n))
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar

    offset = 0  # offset within this rank's shard

    for _ in pbar:
        m = min(n, total_to_make - offset)  # batch size for this iteration (may be < n on the last step)

        # Sample inputs:
        z = torch.randn(m, model.in_channels, latent_size, latent_size, device=device)
        # Enable gradients for z when using feature conditioning (needed for potential gradients)
        if args.use_feature_conditioning:
            z.requires_grad_(True)
        y = y_this_rank[offset : offset + m]
        
        # Override with unconditional class if requested
        if args.use_uncond_class:
            y = torch.full((m,), 1000, device=device, dtype=torch.long)

        assert not args.heun or args.mode == "ode", "Heun's method is only available for ODE sampling."

        # Setup potential for this batch if using feature conditioning
        potential = None
        if args.use_feature_conditioning:
            # Compute features on-the-fly for this batch
            batch_start = start + offset
            batch_end = batch_start + m
            batch_anchor_indices = anchor_indices[batch_start:batch_end]
            batch_anchor_labels = anchor_labels[batch_start:batch_end]
            
            # Compute features for anchor images in this batch
            batch_features_list = []
            batch_anchor_images = []  # Store anchor images for PSNR
            batch_feature_masks = []
            for idx in batch_anchor_indices:
                raw_img, _ = dataset[int(idx)]
                raw_img = raw_img.unsqueeze(0).to(device)
                # Store raw image for PSNR computation
                batch_anchor_images.append(raw_img)
                
                with torch.no_grad():
                    if args.feature_type == 'sit':
                        # Extract SiT features
                        features = extract_sit_features(
                            model, vae, raw_img, args.sit_depth,
                            latents_scale, latents_bias, use_sdvae, args.use_projector,
                            average_features=args.average_features, time=args.time
                        )
                    else:
                        # Extract DINO features
                        raw_img_preprocessed = preprocess_raw_image(raw_img, "dinov2-vit-b")
                        features = extract_encoder_features(encoder, raw_img_preprocessed, "dinov2-vit-b", 
                                                          average_features=args.average_features)
                
                if args.use_pca_feature_mask:
                    # Load precomputed mask from .npz file
                    mask_key = f"mask_{int(idx)}"
                    if pca_masks_data is not None and mask_key in pca_masks_data:
                        feature_mask = torch.from_numpy(pca_masks_data[mask_key]).to(device)
                        # Add batch dimension: (256,) -> (1, 256)
                        feature_mask = feature_mask.unsqueeze(0)
                    else:
                        raise KeyError(f"Mask not found for anchor index {int(idx)} (key: {mask_key})")
                    batch_feature_masks.append(feature_mask)
                batch_features_list.append(features)
            batch_features = torch.cat(batch_features_list, dim=0)  # Shape: (m, num_patches, feature_dim)
            batch_anchor_images = torch.cat(batch_anchor_images, dim=0)  # Shape: (m, 3, H, W)
            batch_feature_masks = torch.cat(batch_feature_masks, dim=0) if len(batch_feature_masks) > 0 else None
            
            # Create potential based on feature averaging
            if args.average_features:
                # Use MeanFeatAlignment for averaged features (DINO or SiT)
                potential = MeanFeatAlignment(
                    cond=batch_features,
                    lamda=args.repa_lambda
                )
            else:
                # Use RepaPotential for spatial feature maps
                potential = RepaPotential(
                    cond=batch_features,
                    lamda=args.repa_lambda,
                    mask=batch_feature_masks
                )
            
            # Use anchor labels when not using unconditional class
            if not args.use_uncond_class:
                # Use the labels from the anchor images
                y = torch.from_numpy(batch_anchor_labels).to(device)

        # Sample images:
        sampling_kwargs = dict(
            model=model,
            latents=z,
            y=y,
            potential=potential,
            num_steps=args.num_steps, 
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
            gibbs=args.gibbs,
            use_projector=args.use_projector if args.use_feature_conditioning and args.feature_type == 'sit' else True,
        )
        
        # Use gradients when feature conditioning is enabled (for potential gradient updates)
        if args.use_feature_conditioning:
            if args.mode == "sde":
                samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":
                samples = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()
        else:
            with torch.no_grad():
                if args.mode == "sde":
                    samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
                elif args.mode == "ode":
                    samples = euler_sampler(**sampling_kwargs).to(torch.float32)
                else:
                    raise NotImplementedError()

        with torch.no_grad():

            samples = vae.decode(denormalize_latents(samples, latents_scale, latents_bias)).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
            ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Compute similarities and PSNR batch-by-batch if enabled
            if args.use_feature_conditioning and args.compute_similarity:
                # Convert samples back to tensor format for feature extraction
                samples_for_features = torch.from_numpy(samples).permute(0, 3, 1, 2).float().to(device)
                
                # Compute PSNR between generated and anchor images
                # samples_for_features is in [0, 255], batch_anchor_images is in [0, 255]
                if args.use_pca_feature_mask:
                    # Compute all PSNRs at once with batched mask interpolation
                    batch_psnr_values = compute_batch_psnr_with_mask(samples_for_features, batch_anchor_images, batch_feature_masks)
                else:
                    # Compute PSNR for each image individually (no mask)
                    batch_psnr_values = []
                    for i in range(m):
                        psnr = compute_psnr(samples_for_features[i:i+1], batch_anchor_images[i:i+1])
                        batch_psnr_values.append(psnr)
                
                # Optionally compute conditioning alignment (features used for generation)
                if args.compute_conditioning_alignment:
                    with torch.no_grad():
                        if args.feature_type == 'sit':
                            # Extract SiT features from generated images
                            gen_features = extract_sit_features(
                                model, vae, samples_for_features, args.sit_depth,
                                latents_scale, latents_bias, use_sdvae, args.use_projector,
                                average_features=args.average_features, time=args.time
                            )
                        else:
                            # Extract DINO features from generated images
                            samples_preprocessed = preprocess_raw_image(samples_for_features, "dinov2-vit-b")
                            gen_features = extract_encoder_features(encoder, samples_preprocessed, "dinov2-vit-b",
                                                                   average_features=args.average_features)
                    
                    # Normalize features at each patch
                    batch_features_cond = batch_features.clone()
                    if args.use_pca_feature_mask:
                        # Apply mask to features
                        batch_features_cond = batch_features_cond * batch_feature_masks.unsqueeze(-1)
                        gen_features = gen_features * batch_feature_masks.unsqueeze(-1)

                    anchor_feat_norm = torch.nn.functional.normalize(batch_features_cond, dim=-1)
                    gen_feat_norm = torch.nn.functional.normalize(gen_features, dim=-1)
                    
                    # Compute cosine similarity patch-wise
                    patch_similarities = (anchor_feat_norm * gen_feat_norm).sum(dim=-1)  # Shape: (m, num_patches)
                    
                    # If features are already averaged, don't average again; otherwise average across patches
                    if args.average_features:
                        batch_conditioning_alignment = patch_similarities.squeeze(-1).cpu().numpy()  # Shape: (m,)
                    else:
                        # Average only over non-masked patches if using PCA mask
                        if args.use_pca_feature_mask:
                            # Sum similarities and divide by number of masked patches
                            masked_sims = patch_similarities * batch_feature_masks  # Zero out unmasked patches
                            sum_sims = masked_sims.sum(dim=-1)  # Sum over patches
                            num_masked = batch_feature_masks.sum(dim=-1)  # Count masked patches
                            batch_conditioning_alignment = (sum_sims / num_masked.clamp(min=1)).cpu().numpy()  # Shape: (m,)
                        else:
                            batch_conditioning_alignment = patch_similarities.mean(dim=-1).cpu().numpy()  # Shape: (m,)
                    
                    # Store conditioning alignment and PSNR with their global indices
                    for i, (align, psnr) in enumerate(zip(batch_conditioning_alignment, batch_psnr_values)):
                        global_idx = start + offset + i
                        conditioning_alignment_list.append((global_idx, align, psnr))
                
                # Compute similarities for all specified backbones
                for backbone_name, additional_encoder in additional_encoders.items():
                    # Compute anchor features for this backbone
                    batch_additional_features_list = []
                    for idx in batch_anchor_indices:
                        raw_img, _ = dataset[int(idx)]
                        raw_img = raw_img.unsqueeze(0).to(device)
                        raw_img_preprocessed = preprocess_raw_image(raw_img, backbone_name)
                        with torch.no_grad():
                            additional_features = extract_encoder_features(additional_encoder, raw_img_preprocessed, backbone_name,
                                                                          average_features=args.average_features)
                        batch_additional_features_list.append(additional_features)
                    batch_additional_features = torch.cat(batch_additional_features_list, dim=0)
                    
                    # Preprocess generated samples for this backbone
                    samples_preprocessed_additional = preprocess_raw_image(samples_for_features, backbone_name)
                    with torch.no_grad():
                        gen_additional_features = extract_encoder_features(additional_encoder, samples_preprocessed_additional, backbone_name,
                                                                          average_features=args.average_features)
                    
                    # Normalize and compute similarities
                    if args.use_pca_feature_mask:
                        batch_additional_features = batch_additional_features * batch_feature_masks.unsqueeze(-1)
                        gen_additional_features = gen_additional_features * batch_feature_masks.unsqueeze(-1)

                    anchor_additional_norm = torch.nn.functional.normalize(batch_additional_features, dim=-1)
                    gen_additional_norm = torch.nn.functional.normalize(gen_additional_features, dim=-1)
                    
                    patch_similarities_additional = (anchor_additional_norm * gen_additional_norm).sum(dim=-1)
                    # If features are already averaged, don't average again; otherwise average across patches
                    if args.average_features:
                        batch_similarities_additional = patch_similarities_additional.squeeze(-1).cpu().numpy()
                    else:
                        # Average only over non-masked patches if using PCA mask
                        if args.use_pca_feature_mask:
                            masked_sims_additional = patch_similarities_additional * batch_feature_masks
                            sum_sims_additional = masked_sims_additional.sum(dim=-1)
                            num_masked = batch_feature_masks.sum(dim=-1)
                            batch_similarities_additional = (sum_sims_additional / num_masked.clamp(min=1)).cpu().numpy()
                        else:
                            batch_similarities_additional = patch_similarities_additional.mean(dim=-1).cpu().numpy()
                    
                    # Store similarities with global indices
                    for i, sim in enumerate(batch_similarities_additional):
                        global_idx = start + offset + i
                        backbone_similarities_lists[backbone_name].append((global_idx, sim))

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = start + offset + i
                Image.fromarray(sample).save(f"{images_dir}/{index:06d}.png")
        offset += m

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    
    # Gather and save feature similarities and PSNR if enabled
    if args.use_feature_conditioning and args.compute_similarity:
        if rank == 0:
            print("Gathering feature similarities and PSNR from all ranks...")
        
        # Gather and save conditioning alignment if enabled
        if args.compute_conditioning_alignment:
            all_conditioning_alignments = [None] * world_size
            dist.all_gather_object(all_conditioning_alignments, conditioning_alignment_list)
            
            if rank == 0:
                # Merge all alignment/PSNR lists
                all_alignments = []
                for align_list in all_conditioning_alignments:
                    all_alignments.extend(align_list)
                
                # Convert to arrays aligned with sample indices
                alignments_dict = {idx: (align, psnr) for idx, align, psnr in all_alignments}
                alignments_array = np.array([alignments_dict.get(i, (np.nan, np.nan))[0] for i in range(args.num_fid_samples)])
                psnr_array = np.array([alignments_dict.get(i, (np.nan, np.nan))[1] for i in range(args.num_fid_samples)])
                
                # Save conditioning alignment
                feature_type_name = args.feature_type if args.feature_type == 'dino' else f'sit_depth{args.sit_depth}'
                alignments_path = os.path.join(sample_folder_dir, f"conditioning_alignment_{feature_type_name}.npy")
                np.save(alignments_path, alignments_array)
                
                # Save PSNR values
                psnr_path = os.path.join(sample_folder_dir, "psnr_values.npy")
                np.save(psnr_path, psnr_array)
                
                # Compute statistics for conditioning alignment
                valid_alignments = alignments_array[~np.isnan(alignments_array)]
                if len(valid_alignments) > 0:
                    mean_align = valid_alignments.mean()
                    std_align = valid_alignments.std()
                    min_align = valid_alignments.min()
                    max_align = valid_alignments.max()
                    
                    print(f"\nConditioning alignment ({args.feature_type}) statistics:")
                    print(f"  Mean: {mean_align:.4f}")
                    print(f"  Std:  {std_align:.4f}")
                    print(f"  Min:  {min_align:.4f}")
                    print(f"  Max:  {max_align:.4f}")
                    print(f"Saved conditioning alignment to {alignments_path}")
                    
                    # Log to MLflow (only mean)
                    if MLFLOW_AVAILABLE and args.use_mlflow:
                        mlflow.log_metrics({
                            f"conditioning_alignment_{args.feature_type}_mean": float(mean_align),
                        })
                        # Log file path instead of uploading large .npy file
                        mlflow.log_param("conditioning_alignment_path", alignments_path)
                else:
                    print(f"Warning: No valid conditioning alignments computed!")
                
                # Compute statistics for PSNR
                valid_psnr = psnr_array[~np.isnan(psnr_array)]
                if len(valid_psnr) > 0:
                    mean_psnr = valid_psnr.mean()
                    std_psnr = valid_psnr.std()
                    min_psnr = valid_psnr.min()
                    max_psnr = valid_psnr.max()
                    
                    print(f"\nPSNR statistics:")
                    print(f"  Mean: {mean_psnr:.2f} dB")
                    print(f"  Std:  {std_psnr:.2f} dB")
                    print(f"  Min:  {min_psnr:.2f} dB")
                    print(f"  Max:  {max_psnr:.2f} dB")
                    print(f"Saved PSNR values to {psnr_path}")
                    
                    # Log to MLflow (only mean)
                    if MLFLOW_AVAILABLE and args.use_mlflow:
                        mlflow.log_metrics({
                            "psnr_mean": float(mean_psnr),
                        })
                        # Log file path instead of uploading large .npy file
                        mlflow.log_param("psnr_values_path", psnr_path)
                else:
                    print(f"Warning: No valid PSNR values computed!")
        
        # Gather and save similarities for all backbone encoders
        for backbone_name in additional_encoders.keys():
            all_backbone_similarities = [None] * world_size
            dist.all_gather_object(all_backbone_similarities, backbone_similarities_lists[backbone_name])
            
            if rank == 0:
                # Merge all similarity lists
                all_backbone_sims = []
                for sim_list in all_backbone_similarities:
                    all_backbone_sims.extend(sim_list)
                
                # Convert to array aligned with sample indices
                backbone_similarities_dict = {idx: sim for idx, sim in all_backbone_sims}
                backbone_similarities_array = np.array([backbone_similarities_dict.get(i, np.nan) for i in range(args.num_fid_samples)])
                
                # Save similarities
                backbone_clean_name = backbone_name.replace('-', '_')
                backbone_similarities_path = os.path.join(sample_folder_dir, f"feature_similarities_{backbone_clean_name}.npy")
                np.save(backbone_similarities_path, backbone_similarities_array)
                
                # Compute statistics
                valid_backbone_similarities = backbone_similarities_array[~np.isnan(backbone_similarities_array)]
                if len(valid_backbone_similarities) > 0:
                    mean_sim = valid_backbone_similarities.mean()
                    std_sim = valid_backbone_similarities.std()
                    min_sim = valid_backbone_similarities.min()
                    max_sim = valid_backbone_similarities.max()
                    
                    print(f"\n{backbone_name} Feature similarity statistics:")
                    print(f"  Mean: {mean_sim:.4f}")
                    print(f"  Std:  {std_sim:.4f}")
                    print(f"  Min:  {min_sim:.4f}")
                    print(f"  Max:  {max_sim:.4f}")
                    print(f"Saved similarities to {backbone_similarities_path}")
                    
                    # Log to MLflow (only mean)
                    if MLFLOW_AVAILABLE and args.use_mlflow:
                        backbone_clean = backbone_name.replace('-', '_')
                        mlflow.log_metrics({
                            f"similarity_{backbone_clean}_mean": float(mean_sim),
                        })
                        # Log file path instead of uploading large .npy file
                        mlflow.log_param(f"similarity_{backbone_clean}_path", backbone_similarities_path)
                else:
                    print(f"Warning: No valid similarities computed for {backbone_name}!")
    
    dist.barrier()
    
    if rank == 0:
        npz_path = create_npz_from_sample_folder(sample_folder_dir, images_dir, args.num_fid_samples)
        
        # Compute evaluation metrics if requested
        if args.compute_metrics and TENSORFLOW_AVAILABLE:
            if not args.ref_batch:
                print("Warning: --ref-batch not provided. Skipping metrics computation.")
            else:
                print("\nComputing evaluation metrics...")
                try:
                    # Initialize TensorFlow session and evaluator
                    config_tf = tf.ConfigProto(allow_soft_placement=True)
                    config_tf.gpu_options.allow_growth = True
                    sess = tf.Session(config=config_tf)
                    evaluator = Evaluator(sess)
                    
                    print("Warming up TensorFlow...")
                    evaluator.warmup()
                    
                    print("Computing reference batch activations...")
                    ref_acts = evaluator.read_activations(args.ref_batch)
                    print("Computing reference batch statistics...")
                    ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)
                    
                    print("Computing sample batch activations...")
                    sample_acts = evaluator.read_activations(npz_path)
                    print("Computing sample batch statistics...")
                    sample_stats, sample_stats_spatial = evaluator.read_statistics(npz_path, sample_acts)
                    
                    metrics = {}
                    
                    # Compute Inception Score
                    try:
                        inception_score = evaluator.compute_inception_score(sample_acts[0])
                        metrics['inception_score'] = float(inception_score)
                        print(f"Inception Score: {inception_score:.4f}")
                    except Exception as e:
                        print(f"Error computing Inception Score: {e}")
                    
                    # Compute FID
                    try:
                        fid = sample_stats.frechet_distance(ref_stats)
                        metrics['fid'] = float(fid)
                        print(f"FID: {fid:.4f}")
                    except Exception as e:
                        print(f"Error computing FID: {e}")
                    
                    # Compute sFID
                    try:
                        sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)
                        metrics['sfid'] = float(sfid)
                        print(f"sFID: {sfid:.4f}")
                    except Exception as e:
                        print(f"Error computing sFID: {e}")
                    
                    # Compute Precision/Recall
                    try:
                        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
                        metrics['precision'] = float(prec)
                        metrics['recall'] = float(recall)
                        print(f"Precision: {prec:.4f}")
                        print(f"Recall: {recall:.4f}")
                    except Exception as e:
                        print(f"Error computing Precision/Recall: {e}")
                    
                    # Save metrics to file
                    metrics_file = os.path.join(sample_folder_dir, "metrics.json")
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    print(f"\nMetrics saved to {metrics_file}")
                    
                    # Log metrics to MLflow
                    if MLFLOW_AVAILABLE and args.use_mlflow:
                        mlflow.log_metrics(metrics)
                        mlflow.log_artifact(metrics_file)
                        print("Metrics logged to MLflow")
                    
                    # Close TensorFlow session
                    sess.close()
                    
                except Exception as e:
                    print(f"Error during metrics computation: {e}")
                    import traceback
                    traceback.print_exc()
        elif args.compute_metrics and not TENSORFLOW_AVAILABLE:
            print("Warning: TensorFlow not available. Cannot compute metrics.")
    
    # Synchronize all ranks after FID computation (rank 0 may take a long time)
    dist.barrier()
    
    if rank == 0:
        # Log final artifacts to MLflow
        if MLFLOW_AVAILABLE and args.use_mlflow:
            # Log the config YAML (small file, safe to upload)
            config_yaml_path = os.path.join(sample_folder_dir, "config.yaml")
            if os.path.exists(config_yaml_path):
                mlflow.log_artifact(config_yaml_path)
            
            # Log paths to large files instead of uploading them
            if os.path.exists(npz_path):
                mlflow.log_param("npz_path", npz_path)
            
            # Log sample directory path
            mlflow.log_param("output_dir", sample_folder_dir)
            
            # End the run
            mlflow.end_run()
            print("MLflow run completed.")
        
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed params
    parser.add_argument("--global-seed", type=int, default=0)

    # precision params
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving params
    parser.add_argument("--sample-dir", type=str, default="samples")

    # ckpt params
    parser.add_argument("--exp-path", type=str, default=None, help="Path to the specific experiment directory.")
    parser.add_argument("--train-steps", type=str, default=None, help="The checkpoint of the model to sample from.")

    # number of samples
    parser.add_argument("--pproc-batch-size", type=int, default=128)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    
    # feature conditioning params
    parser.add_argument("--use-feature-conditioning", action="store_true", default=False,
                       help="Enable feature-conditioned generation using REPA potential.")
    parser.add_argument("--feature-type", type=str, default="dino", choices=["dino", "sit"],
                       help="Type of features to use for conditioning: 'dino' (DINOv2) or 'sit' (SiT internal features).")
    parser.add_argument("--sit-depth", type=int, default=8,
                       help="Which SiT layer depth to extract features from (only used when feature-type=sit).")
    parser.add_argument("--use-projector", action="store_true", default=False,
                       help="Use projected SiT features instead of raw features (only used when feature-type=sit).")
    parser.add_argument("--average-features", action="store_true", default=False,
                       help="Average features spatially and use MeanFeatAlignment potential. "
                            "If False, uses full spatial feature maps with RepaPotential. "
                            "Works with both DINO and SiT features.")
    parser.add_argument("--compute-similarity", action="store_true", default=False,
                       help="Compute and save cosine similarity between anchor and generated image features. "
                            "Only effective when --use-feature-conditioning is enabled.")
    parser.add_argument("--compute-conditioning-alignment", action="store_true", default=False,
                       help="Compute alignment between generated images and conditioning features (DINO or SiT). "
                            "Only effective when --compute-similarity is enabled. If False, only computes similarities for additional backbones.")
    parser.add_argument("--additional-similarity-backbones", type=str, default=None,
                       help="Comma-separated list of additional encoder backbones for computing similarities. "
                            "Examples: 'clip-vit-b', 'mocov3-vit-b', 'jepa-vit-h', 'mae-vit-l'. "
                            "Only effective when --compute-similarity is enabled.")
    parser.add_argument("--repa-lambda", type=float, default=5000.0,
                       help="Lambda parameter for REPA potential (feature guidance strength).")
    parser.add_argument("--anchor-seed", type=int, default=42,
                       help="Seed for selecting anchor images from dataset. Used to ensure reproducibility across runs.")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing dataset for anchor images (ImageNet H5 or COCO root).")
    parser.add_argument("--dataset-type", type=str, default="imagenet", choices=["imagenet", "coco", "coco-npz"],
                       help="Type of dataset to use for anchor images: 'imagenet' (H5) or 'coco' (SimpleCOCO).")
    parser.add_argument("--coco-split", type=str, default="train", choices=["train", "val", "test"],
                       help="COCO split to use when dataset-type=coco (default: train).")
    parser.add_argument("--coco-year", type=str, default="2017", choices=["2014", "2017"],
                       help="COCO year to use when dataset-type=coco (default: 2017).")
    parser.add_argument("--use-uncond-class", action="store_true", default=False,
                       help="Use unconditional class (1000) for generation, disabling label conditioning. "
                            "Works with or without feature conditioning. "
                            "If False, uses the class labels from label-sampling strategy (or anchor labels with feature conditioning).")

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine", "reverse_linear"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False,
                        help="Use Heun's method for ODE sampling.")
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)
    parser.add_argument("--gibbs", action=argparse.BooleanOptionalAction, default=True,
                        help="Use Gibbs sampling method.")
    parser.add_argument("--time", type=float, default=0.0,
                        help="Time parameter for SiT feature extraction (only used when feature-type=sit).")

    parser.add_argument(
        "--label-sampling",
        type=str,
        choices=["random", "equal"],
        default="equal",
        help="Choose how to sample class labels when generating images.",
    )
    
    parser.add_argument(
        "--use_pca_feature_mask", action="store_true", default=False, 
        help="Use PCA to create a binary mask for feature conditioning. Only effective when --use-feature-conditioning is enabled."
    )
    
    # MLflow logging params
    parser.add_argument("--use-mlflow", action="store_true", default=False,
                       help="Enable MLflow experiment tracking and logging.")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None,
                       help="MLflow tracking URI (e.g., './mlruns', 'file:///path/to/mlruns', or remote server URL). "
                            "If not specified, uses default './mlruns' directory.")
    parser.add_argument("--mlflow-experiment-name", type=str, default="repa-generation",
                       help="MLflow experiment name.")
    parser.add_argument("--mlflow-run-name", type=str, default=None,
                       help="MLflow run name. If None, will be auto-generated.")
    
    # Evaluation metrics params
    parser.add_argument("--compute-metrics", action="store_true", default=False,
                       help="Compute FID, IS, sFID, Precision/Recall metrics after generation.")
    parser.add_argument("--ref-batch", type=str, default=None,
                       help="Path to reference batch npz file for computing FID and other metrics.")

    args = parser.parse_args()
    main(args)
