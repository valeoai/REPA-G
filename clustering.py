"""
Efficient K-means clustering for large-scale features.

This script performs K-means clustering on millions of high-dimensional features
stored in .pt files, using FAISS GPU acceleration for maximum speed.

FAISS provides orders of magnitude faster clustering than pure PyTorch implementations
and is optimized for billion-scale datasets.

Requirements:
    pip install faiss-gpu  # or faiss-cpu if no GPU available

Example usage:
    python clustering.py \\
        --features-path data/dinov2_features.pt \\
        --output-path data/clusters_k100.pt \\
        --n-clusters 100 \\
        --max-iters 300 \\
        --device cuda:0
"""

import argparse
import torch
import numpy as np
import faiss
from pathlib import Path
import time
import os


def load_features(features_path):
    """Load features from .pt file saved by extract_features.py.
    
    Expected format:
        {
            'features': Tensor of shape [N, D],
            'labels': Tensor of shape [N],
            'feature_dim': int,
            'num_samples': int,
            'feature_type': str,
            ... (additional metadata)
        }
    """
    print(f"Loading features from {features_path}...")
    data = torch.load(features_path, map_location='cpu')
    
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected dictionary format from extract_features.py, got {type(data)}. "
            f"Please use extract_features.py to generate features."
        )
    
    if 'features' not in data:
        raise ValueError(
            f"Missing 'features' key in loaded data. Available keys: {list(data.keys())}. "
            f"Please use extract_features.py to generate features."
        )
    
    features = data['features']
    
    # Ensure features are 2D [N, D]
    if features.ndim > 2:
        original_shape = features.shape
        features = features.reshape(-1, features.shape[-1])
        print(f"Reshaped features from {original_shape} to {features.shape}")
    
    # Print metadata
    print(f"Loaded {features.shape[0]} features of dimension {features.shape[1]}")
    if 'feature_type' in data:
        print(f"Feature type: {data['feature_type']}")
    if 'encoder_type' in data:
        print(f"Encoder type: {data['encoder_type']}")
    if 'sit_model' in data and 'sit_depth' in data:
        print(f"SiT model: {data['sit_model']}, depth: {data['sit_depth']}")
    
    return features


def kmeans_faiss(features, n_clusters, max_iters=300, gpu_id=0, verbose=True, seed=42, force_cpu_train=False):
    """Perform K-means clustering using FAISS (ultra-fast GPU implementation).
    
    Args:
        features: [N, D] numpy array of features (float32, L2-normalized)
        n_clusters: number of clusters
        max_iters: maximum number of iterations
        gpu_id: GPU device ID (-1 for CPU)
        verbose: whether to print progress
        seed: random seed
        force_cpu_train: if True, train on CPU but use GPU for inference
    
    Returns:
        centroids: [n_clusters, D] numpy array of cluster centers
        assignments: [N] numpy array of cluster assignments
        inertia: final within-cluster sum of squares
    """
    n_samples, d = features.shape
    
    # Ensure float32 and C-contiguous for FAISS
    if features.dtype != np.float32:
        features = features.astype(np.float32)
    if not features.flags['C_CONTIGUOUS']:
        features = np.ascontiguousarray(features)
    
    # Re-normalize features to ensure perfect unit norm (sometimes needed for numerical stability)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norms + 1e-10)
    
    train_on_gpu = gpu_id >= 0 and not force_cpu_train
    inference_on_gpu = gpu_id >= 0 and not force_cpu_train
    
    print(f"Running FAISS K-means: Training on {'GPU' if train_on_gpu else 'CPU'}, Inference on {'GPU' if inference_on_gpu else 'CPU'}")
    print(f"Features shape: {features.shape}, dtype: {features.dtype}")
    start_time = time.time()
    
    # Use standard Kmeans wrapper which is more robust
    kmeans = faiss.Kmeans(
        d=d,
        k=n_clusters,
        niter=max_iters,
        nredo=5,  # Try 5 different initializations
        verbose=verbose,
        seed=seed,
        spherical=True,  # Spherical k-means (cosine)
        gpu=train_on_gpu,  # Training device
    )
    
    # Perform clustering (training)
    kmeans.train(features)
    
    # Get centroids
    centroids = kmeans.centroids
    
    # For inference, potentially use GPU even if training was on CPU
    if inference_on_gpu and not train_on_gpu:
        print("Moving index to GPU for faster inference...")
        # Create GPU index for inference
        res = faiss.StandardGpuResources()
        
        cpu_index = faiss.IndexFlatIP(d)
        cpu_index.add(centroids)
        
        # Transfer to GPU (correct signature: no config parameter)
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
        
        # Assign all points to clusters using GPU
        print("Assigning clusters to all points on GPU...")
        distances, assignments = gpu_index.search(features, 1)
    else:
        # Use the kmeans index (either GPU or CPU depending on train_on_gpu)
        print("Assigning clusters to all points...")
        distances, assignments = kmeans.index.search(features, 1)
    
    assignments = assignments.ravel()
    
    # Compute inertia (sum of squared distances)
    inertia = (distances ** 2).sum()
    
    elapsed = time.time() - start_time
    print(f"K-means completed in {elapsed:.2f}s")
    
    return centroids, assignments, inertia


def main():
    parser = argparse.ArgumentParser(description='K-means clustering for large-scale features')
    parser.add_argument('--features-path', type=str, required=True,
                        help='Path to input .pt file containing features')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save clustering results (.pt file)')
    parser.add_argument('--n-clusters', type=int, required=True,
                        help='Number of clusters (K)')
    parser.add_argument('--max-iters', type=int, default=300,
                        help='Maximum number of iterations (default: 300)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: cluster only first 10000 features')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force all clustering operations on CPU (training and inference) for maximum stability')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check device availability
    use_gpu = args.device.startswith('cuda') and torch.cuda.is_available()
    if args.force_cpu:
        print("Forcing CPU for all clustering operations (training and inference)")
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
        use_gpu = False
    
    # Extract GPU ID if specified
    gpu_id = 0
    if use_gpu and ':' in args.device:
        gpu_id = int(args.device.split(':')[1])
    elif not use_gpu:
        gpu_id = -1
    
    print(f"Using device: {args.device}")
    
    # Load features
    features = load_features(args.features_path)
    
    # Debug mode: use random subset
    if args.debug:
        subset_size = min(10000, len(features))
        # Random sampling instead of first N
        indices = torch.randperm(len(features))[:subset_size]
        features = features[indices]
        print(f"\nDebug mode: using random subset of {len(features)} features")
    
    # Convert to numpy for diagnostics and FAISS (features already L2-normalized during extraction)
    features_np = features.numpy().astype(np.float32)

    # Check feature variance
    print("Mean feature std:", features_np.std(axis=0).mean())

    # Check norm distribution
    norms = np.linalg.norm(features_np, axis=1)
    print("Norm mean:", norms.mean(), "std:", norms.std())

    # Check pairwise cosine similarity (sample)
    idx = np.random.choice(len(features_np), min(100, len(features_np)), replace=False)
    cos = features_np[idx] @ features_np[idx].T
    print("Cosine mean off-diag:", (cos.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
    
    # Run K-means with FAISS
    print(f"\nRunning K-means with {args.n_clusters} clusters...")
    start_time = time.time()
    
    centroids, assignments, inertia = kmeans_faiss(
        features=features_np,
        n_clusters=args.n_clusters,
        max_iters=args.max_iters,
        gpu_id=gpu_id,
        verbose=True,
        seed=args.seed,
        force_cpu_train=args.force_cpu
    )
    
    # Convert back to torch
    centroids = torch.from_numpy(centroids)
    assignments = torch.from_numpy(assignments).long()
    
    total_time = time.time() - start_time
    print(f"\nK-means completed in {total_time:.2f}s")
    print(f"Final inertia: {inertia:.2f}")
    print(f"\nK-means completed in {total_time:.2f}s")
    print(f"Final inertia: {inertia:.2f}")
    
    # Compute cluster statistics
    unique, counts = torch.unique(assignments, return_counts=True)
    print(f"\nCluster statistics:")
    print(f"  Min cluster size: {counts.min().item()}")
    print(f"  Max cluster size: {counts.max().item()}")
    print(f"  Mean cluster size: {counts.float().mean().item():.1f}")
    print(f"  Std cluster size: {counts.float().std().item():.1f}")
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'centroids': centroids,
        'assignments': assignments,
        'inertia': inertia,
        'n_clusters': args.n_clusters,
        'n_samples': len(features),
        'feature_dim': features.shape[1],
        'cluster_sizes': counts,
        'args': vars(args)
    }
    
    torch.save(save_dict, output_path)
    print(f"\nResults saved to {output_path}")
    print(f"  Centroids shape: {centroids.shape}")
    print(f"  Assignments shape: {assignments.shape}")


if __name__ == '__main__':
    main()    