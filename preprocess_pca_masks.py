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
import os
import json 

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from dataset import CustomINH5PathDataset
from dataset_coco import SimpleCOCODataset
from torchvision import transforms
from utils import load_encoders, compute_mask_from_pca
from train_repae import preprocess_raw_image

def extract_encoder_features(encoder, raw_image, encoder_type):
    """Extract features from encoder with proper postprocessing.
    
    Args:
        encoder: Vision encoder model
        raw_image: Preprocessed image tensor
        encoder_type: Type of encoder (e.g., 'dinov2', 'mocov3', 'clip')
    
    Returns:
        Extracted features tensor
    """
    z = encoder.forward_features(raw_image)
    if 'mocov3' in encoder_type:
        z = z[:, 1:]  # Remove CLS token
    elif 'dinov2' in encoder_type:
        z = z['x_norm_patchtokens']
    return z

def main(args):
    """
    Pre-compute and cache PCA masks for images specified in anchor_indices_50000_seed42.npy.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "This script requires at least one GPU"
    
    # We don't need gradients for this script
    torch.set_grad_enabled(False)

    # Setup DDP
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    device = f"cuda:{device_id}"
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device_id)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Load anchor indices
    if rank == 0:
        print(f"Loading anchor indices from {args.anchor_file}")
    anchor_indices = np.load(args.anchor_file)
    num_samples = len(anchor_indices)
    if rank == 0:
        print(f"Loaded {num_samples} anchor indices")

    # Load encoder
    encoder_type = "dinov2-vit-b" # Hardcoded as this is for DINOv2 PCA masks
    if rank == 0:
        print(f"Loading {encoder_type} encoder...")
        encoders_list, _, _ = load_encoders(encoder_type, device, args.resolution)
        encoder = encoders_list[0]
        encoder.eval()

    dist.barrier()

    if rank != 0:
        encoders_list, _, _ = load_encoders(encoder_type, device, args.resolution)
        encoder = encoders_list[0]
        encoder.eval()

    # Load full dataset based on dataset type
    if args.dataset_type == "imagenet":
        full_dataset = CustomINH5PathDataset(args.data_dir)
    elif args.dataset_type == "coco":
        if rank == 0:
            print(f"Loading COCO {args.coco_split}{args.coco_year} dataset...")
        full_dataset = SimpleCOCODataset(
            root=args.data_dir,
            split=args.coco_split,
            year=args.coco_year,
            image_size=args.resolution,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.PILToTensor(),
            ])
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    # Create subset dataset with only anchor indices
    subset_indices = anchor_indices.tolist()
    subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    
    # Create a custom dataset that also returns the index
    class IndexedDataset(torch.utils.data.Dataset):
        def __init__(self, subset_dataset, subset_indices, dataset_type):
            self.subset_dataset = subset_dataset
            self.subset_indices = subset_indices
            self.dataset_type = dataset_type
        
        def __len__(self):
            return len(self.subset_dataset)
        
        def __getitem__(self, idx):
            data = self.subset_dataset[idx]
            if self.dataset_type == "imagenet":
                img, label, fname = data
            elif self.dataset_type == "coco":
                img, fname = data
                label = 0  # Dummy label for COCO
            # Return the original anchor index along with the data
            return img, label, fname, self.subset_indices[idx]
    
    indexed_dataset = IndexedDataset(subset_dataset, subset_indices, args.dataset_type)
    
    sampler = DistributedSampler(
        indexed_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    dataloader = DataLoader(
        indexed_dataset,
        batch_size=args.pproc_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    if rank == 0:
        print(f"Processing {len(indexed_dataset)} images across {world_size} GPUs")

    dist.barrier()

    # Collect all masks with their corresponding anchor indices
    local_masks = []
    local_indices = []
    
    pbar = tqdm(dataloader, desc="Computing PCA Masks", disable=(rank != 0))

    for batch_images, _, batch_fnames, batch_anchor_indices in pbar:
        batch_images = batch_images.to(device)

        # Preprocess images for the encoder
        raw_img_preprocessed = preprocess_raw_image(batch_images, encoder_type)
        
        # Extract features
        features = extract_encoder_features(encoder, raw_img_preprocessed, encoder_type)
        
        # Compute PCA masks for the batch
        feature_masks = compute_mask_from_pca(features).cpu().numpy()
        
        # Store masks with their corresponding anchor indices
        for i in range(len(batch_images)):
            local_masks.append(feature_masks[i])
            local_indices.append(batch_anchor_indices[i].item())
    
    dist.barrier()
    
    # Gather all masks and indices from all ranks
    all_masks_gathered = [None] * world_size
    all_indices_gathered = [None] * world_size
    dist.all_gather_object(all_masks_gathered, local_masks)
    dist.all_gather_object(all_indices_gathered, local_indices)
    
    if rank == 0:
        # Flatten the gathered lists
        all_masks_flat = []
        all_indices_flat = []
        for masks_list, indices_list in zip(all_masks_gathered, all_indices_gathered):
            all_masks_flat.extend(masks_list)
            all_indices_flat.extend(indices_list)
        
        # Create a dictionary mapping anchor indices to masks
        masks_dict = {}
        for anchor_idx, mask in zip(all_indices_flat, all_masks_flat):
            masks_dict[f"mask_{anchor_idx}"] = mask
        
        # Construct output filename from anchor filename
        # Convert anchor_indices_50000_seed42.npy -> pca_masks_50000_seed42.npz
        anchor_basename = os.path.basename(args.anchor_file)
        output_basename = anchor_basename.replace("anchor_indices_", "pca_masks_").replace(".npy", ".npz")
        output_dir = os.path.dirname(args.anchor_file) if args.output_dir is None else args.output_dir
        output_path = os.path.join(output_dir, output_basename)
        
        np.savez(output_path, **masks_dict)
        print(f"Saved {len(masks_dict)} PCA masks to: {output_path}")
        print(f"Masks can be loaded with: data = np.load('{output_path}'); mask = data['mask_<anchor_idx>']")
        
    if rank == 0:
        print("Done.")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--data-dir", type=str, default="/datasets_local/nsereyjo/ImageNet/",
                       help="Directory containing dataset (ImageNet H5 or COCO root).")
    parser.add_argument("--dataset-type", type=str, default="imagenet", choices=["imagenet", "coco"],
                       help="Dataset type: imagenet or coco")
    parser.add_argument("--coco-split", type=str, default="train", choices=["train", "val"],
                       help="COCO split (train or val)")
    parser.add_argument("--coco-year", type=str, default="2017", choices=["2014", "2017"],
                       help="COCO year (2014 or 2017)")
    parser.add_argument("--anchor-file", type=str, required=True,
                       help="Path to the anchor indices .npy file (e.g., log/samples/anchor_indices_50000_seed42.npy).")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for the pca_masks .npz file. If None, saves next to anchor file.")
    parser.add_argument("--resolution", type=int, default=256, help="Image resolution.")
    parser.add_argument("--pproc-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()
    main(args)
