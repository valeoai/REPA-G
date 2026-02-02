import torch
import os
from dictdot import dictdot
import json
import numpy as np
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
import torchvision.transforms.functional as TF

from dataset import CustomINH5Dataset
from models.sit import SiT_models
from models.autoencoder import vae_models
from samplers import get_score_from_velocity, compute_diffusion
from potential import feature_dir_update, RepaPotential, MeanFeatAlignment
from utils import load_encoders, normalize_latents, load_sit_and_vae
from train_repae import preprocess_raw_image
import matplotlib as mpl


mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "cm",   # Computer Modern for math
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
})
try:
    from diffusers import AutoencoderKL
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


def extract_dino_features(encoder, encoder_type, images, resolution=256, average_features=True):
    """Extract DINOv2 spatial features and optionally average spatially."""
    if 'dinov2' in encoder_type or 'dinov1' in encoder_type:
        x = images / 255.0
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        if x.shape[-1] == resolution:
            target_size = 224 * (resolution // 256)
            x = F.interpolate(x, size=target_size, mode='bicubic', align_corners=False)
    
    with torch.no_grad():
        z = encoder.forward_features(x)
        if 'dinov2' in encoder_type or 'dinov1' in encoder_type:
            z = z['x_norm_patchtokens']
        
        if average_features:
            z_avg = z.mean(dim=1)  # Average spatially
            return z_avg
        else:
            return z  # Return full spatial map [B, num_patches, D]


def extract_sit_features(model, vae, images, depth, latents_stats, use_sdvae=False, use_projector=False, average_features=True):
    """Extract SiT internal activations and optionally average spatially."""
    with torch.no_grad():
        images_norm = images.float() / 127.5 - 1.0
        
        if use_sdvae:
            posterior = vae.encode(images_norm).latent_dist
            z = posterior.sample()
        else:
            posterior = vae.encode(images_norm)
            z = posterior.sample()
        
        z = normalize_latents(z, latents_stats['scale'], latents_stats['bias'])
        
        B = z.shape[0]
        device = z.device
        t = torch.zeros(B, device=device)
        y = torch.full((B,), 1000, dtype=torch.long, device=device)
        
        feats = model.forward_feats(z, t, y, depth=depth, use_projector=use_projector)
        
        # When use_projector=True, feats is a list; when False, it's a tensor
        if use_projector:
            feats = feats[0]  # Take first projector output
        
        if average_features:
            feats_avg = feats.mean(dim=1)  # Average spatially
            return feats_avg
        else:
            return feats

def euler_maruyama_sampler(
        model,
        latents,
        y,
        num_steps=20,
        potential=None,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        use_projector=True,
        gibbs=False,
    ):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
            
    _dtype = latents.dtype
    
    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
        dt = t_next - t_cur
        x_cur = x_next
        if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
            model_input = torch.cat([x_cur] * 2, dim=0)
            y_cur = torch.cat([y, y_null], dim=0)
        else:
            model_input = x_cur
            y_cur = y            
        kwargs = dict(y=y_cur)
        time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
        diffusion = compute_diffusion(t_cur)            
        eps_i = torch.randn_like(x_cur).to(device)
        deps = eps_i * torch.sqrt(torch.abs(dt))

        # compute drift
        if potential is not None and t_cur <= guidance_high and t_cur >= guidance_low:
            v_cur, zs = model.inference_feats(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), use_projector=use_projector, **kwargs)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur
            d_feat = feature_dir_update(model_input, zs[0], potential, retain_graph=True).to(torch.float64)
            if gibbs:
                d_feat*=t_cur
            d_cur = d_cur.to(torch.float64) + d_feat
        else:
            v_cur = model.inference(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
            ).to(torch.float64)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur
        if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

        x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y            
    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
        ) * t_cur
    
    # compute drift
    if potential is not None and t_cur <= guidance_high and t_cur >= guidance_low:
        v_cur, zs = model.inference_feats(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), use_projector=use_projector, **kwargs)
        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur
        d_feat = feature_dir_update(model_input, zs[0], potential, retain_graph=True).to(torch.float64)
        if gibbs:
            d_feat*=t_cur
        d_cur = d_cur.to(torch.float64) + d_feat
    else:
        v_cur = model.inference(
            model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
        ).to(torch.float64)
        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur
    _, zs = model.inference_feats(model_input.to(dtype=_dtype), 0*time_input.to(dtype=_dtype), use_projector=use_projector, **kwargs)
    return zs[0] / torch.linalg.norm(zs[0], ord=2, dim=-1, keepdim=True)

def symetrized_kl(model, phi_1, phi_2, y=1000, mc_sample=1, num_steps=20, device='cuda', lamda=30000, feature_type='dino', use_projector=True, average_features=True, gibbs=False):
    # Normalize features before passing to potential
    phi_1_norm = phi_1 / torch.linalg.norm(phi_1, ord=2, dim=-1, keepdim=True)
    phi_2_norm = phi_2 / torch.linalg.norm(phi_2, ord=2, dim=-1, keepdim=True)
    
    # Use MeanFeatAlignment for averaged features, RepaPotential for spatial maps
    if average_features:
        potential_1 = MeanFeatAlignment(phi_1_norm, lamda=lamda)
        potential_2 = MeanFeatAlignment(phi_2_norm, lamda=lamda)
    else:
        potential_1 = RepaPotential(phi_1_norm, lamda=lamda)
        potential_2 = RepaPotential(phi_2_norm, lamda=lamda)
    z1 = torch.randn(mc_sample, model.in_channels, 32, 32, device=device, requires_grad=True)  
    z2 = torch.randn(mc_sample, model.in_channels, 32, 32, device=device, requires_grad=True)  
    y = torch.tensor([y], device=device).to(torch.long)  
    f1 = euler_maruyama_sampler(model,z1,y,potential=potential_1,num_steps=num_steps, use_projector=use_projector, gibbs=gibbs)
    f2 = euler_maruyama_sampler(model,z2,y,potential=potential_2,num_steps=num_steps, use_projector=use_projector, gibbs=gibbs)
    
    return ((f1.mean(dim=0)-f2.mean(dim=0))*(phi_1_norm - phi_2_norm)).sum()


def analyze_mc_variance(model, phi_1, phi_2, max_mc_samples, num_repeats, num_steps, device, lamda, feature_type, use_projector, average_features=True):
    """
    Analyze how KL divergence variance evolves with number of MC samples.
    
    Args:
        model: SiT model
        phi_1, phi_2: Feature vectors to compare
        max_mc_samples: Maximum number of MC samples to test
        num_repeats: Number of times to repeat computation for each k
        num_steps: Number of diffusion steps
        device: Device to use
        lamda: Lambda parameter for potential
        feature_type: Type of features used
        use_projector: Whether to use projector
        average_features: Whether features are spatially averaged (use MeanFeatAlignment) or spatial maps (use RepaPotential)
    
    Returns:
        mc_samples_list: List of k values tested
        mean_estimates: Mean KL estimate for each k
        std_estimates: Standard deviation for each k
    """
    # Test k values: 1, 2, 5, 10, 20, 50, ...
    mc_samples_list = [1, 2, 5, 10, 20, 50, 100]
    mc_samples_list = [k for k in mc_samples_list if k <= max_mc_samples]
    if max_mc_samples not in mc_samples_list and max_mc_samples > mc_samples_list[-1]:
        mc_samples_list.append(max_mc_samples)
    
    mean_estimates = []
    std_estimates = []
    
    print(f"\nAnalyzing MC variance with up to {max_mc_samples} samples...")
    for k in tqdm(mc_samples_list, desc="Testing MC samples"):
        kl_values = []
        for _ in range(num_repeats):
            kl = symetrized_kl(model, phi_1, phi_2, mc_sample=k, num_steps=num_steps, 
                             device=device, lamda=lamda, feature_type=feature_type, use_projector=use_projector,
                             average_features=average_features)
            kl_values.append(kl.item())
        
        mean_estimates.append(np.mean(kl_values))
        std_estimates.append(np.std(kl_values))
    
    return mc_samples_list, mean_estimates, std_estimates


def visualize_mc_variance(mc_samples_list, mean_estimates, std_estimates, save_path="mc_variance_analysis.pdf"):
    """
    Visualize how variance evolves with number of MC samples.
    
    Creates two subplots:
    1. Mean estimate and confidence interval vs k
    2. Standard deviation vs k (log-log scale)
    """
    mc_samples = np.array(mc_samples_list)
    means = np.array(mean_estimates)
    stds = np.array(std_estimates)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean estimate with confidence intervals
    ax1.plot(mc_samples, means, 'b-o', linewidth=2, markersize=8, label='Mean estimate')
    ax1.fill_between(mc_samples, means - stds, means + stds, alpha=0.3, color='blue', label='±1 std')
    ax1.set_xlabel('Number of MC samples (k)', fontsize=12)
    ax1.set_ylabel('KL Divergence Estimate', fontsize=12)
    ax1.set_title('KL Estimate Convergence', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Standard deviation (log-log scale)
    ax2.plot(mc_samples, stds, 'r-s', linewidth=2, markersize=8, label='Standard deviation')
    
    # Fit and plot 1/sqrt(k) reference line
    if len(mc_samples) > 1:
        # Use first point to calibrate the reference line
        c = stds[0] * np.sqrt(mc_samples[0])
        reference = c / np.sqrt(mc_samples)
        ax2.plot(mc_samples, reference, 'k--', linewidth=1.5, alpha=0.7, label=r'$\propto 1/\sqrt{k}$')
    
    ax2.set_xlabel('Number of MC samples (k)', fontsize=12)
    ax2.set_ylabel('Standard Deviation', fontsize=12)
    ax2.set_title('Variance Reduction with MC Samples', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nMC variance analysis saved to {save_path}")
    print(f"\nSummary:")
    print(f"  k={mc_samples[0]:3d}: mean={means[0]:.4f}, std={stds[0]:.4f} (CV={stds[0]/means[0]*100:.1f}%)")
    print(f"  k={mc_samples[-1]:3d}: mean={means[-1]:.4f}, std={stds[-1]:.4f} (CV={stds[-1]/means[-1]*100:.1f}%)")
    print(f"  Variance reduction: {stds[0]/stds[-1]:.2f}x")

def visu_paper(embedding_dino, save_path="embedding_dino.pdf"):
    """ visualization of DINO embeddings."""
    
    emb = np.asarray(embedding_dino)
    x, y = emb[:, 0], emb[:, 1]
    
    slopes = y / x
    a = np.min(slopes)
    b = np.max(slopes)
    print("B/A ratio:", b / a)
    t = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)

    fig, ax = plt.subplots(figsize=(3.25, 2.5))


    line_max, = ax.plot(
        t, b * t,
        linestyle="--",
        linewidth=1,
        color="black"
    )

    line_min, = ax.plot(
        t, a * t,
        linestyle=":",
        linewidth=1,
        color="black"
    )

    sc = ax.scatter(
        x, y,
        s=8,
        alpha=0.6,
        linewidths=0,
        color="black"
    )
    ax.legend(
        [line_max, line_min],
        ["max slope", "min slope"],
        frameon=False,
        fontsize=8,
        loc="upper left"
    )


    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Squared Feature Distance")
    ax.set_ylabel("Symmetrized KL Divergence")
    
    ax.text(
    0.7, 0.02,
    rf"$B/A = {b/a:.2f}$",
    transform=ax.transAxes,
    fontsize=8,
    va="bottom",
    ha="left"
    )
    
    ax.legend(
    ["max slope", "min slope"],
    frameon=False,
    loc="upper left"
    )

    # Minimalist axis style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(direction="out", length=3, width=0.8)

    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}")
    
    # Save embeddings to .npy files
    base_path = os.path.splitext(save_path)[0]
    np.save(f"{base_path}_dinov2.npy", emb)
    print(f"Saved embeddings to {base_path}_dinov2.npy")

    
    
def inequality_visualization(embeddings_dino, embeddings_sit, save_path="embedding_inequality.png"):
    """Visualize embeddings inequality with two subplots for DINO and SiT."""
    emb_dino = np.array(embeddings_dino)
    x_dino = emb_dino[:, 0]
    y_dino = emb_dino[:, 1]
    
    # Filter out invalid values (zeros, NaNs, infs)
    valid_mask_dino = (x_dino > 0) & np.isfinite(x_dino) & np.isfinite(y_dino) & (y_dino > 0)
    x_dino_valid = x_dino[valid_mask_dino]
    y_dino_valid = y_dino[valid_mask_dino]
    
    # Remove outliers using IQR method on y values
    if len(y_dino_valid) > 10:
        q1, q3 = np.percentile(y_dino_valid, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        outlier_mask = (y_dino_valid >= lower_bound) & (y_dino_valid <= upper_bound)
        x_dino_valid = x_dino_valid[outlier_mask]
        y_dino_valid = y_dino_valid[outlier_mask]
        print(f"DINO: Filtered {(~outlier_mask).sum()} outliers from {len(outlier_mask)} points")
    
    if len(x_dino_valid) > 0:
        a_dino = y_dino_valid / x_dino_valid
        # Filter out any remaining NaN/inf from the division
        a_dino = a_dino[np.isfinite(a_dino)]
        if len(a_dino) > 0:
            B_dino = np.max(a_dino)
            A_dino = np.min(a_dino)
        else:
            B_dino = A_dino = 1.0  # Fallback
    else:
        B_dino = A_dino = 1.0  # Fallback
    
    emb_sit = np.array(embeddings_sit)
    x_sit = emb_sit[:, 0]
    y_sit = emb_sit[:, 1]
    
    # Filter out invalid values (zeros, NaNs, infs) and extreme outliers
    valid_mask_sit = (x_sit > 0) & np.isfinite(x_sit) & np.isfinite(y_sit) & (y_sit > 0)
    x_sit_valid = x_sit[valid_mask_sit]
    y_sit_valid = y_sit[valid_mask_sit]
    
    # Remove outliers using IQR method on y values
    if len(y_sit_valid) > 10:
        q1, q3 = np.percentile(y_sit_valid, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr  # Use 3*IQR for outlier detection
        upper_bound = q3 + 3 * iqr
        outlier_mask = (y_sit_valid >= lower_bound) & (y_sit_valid <= upper_bound)
        x_sit_valid = x_sit_valid[outlier_mask]
        y_sit_valid = y_sit_valid[outlier_mask]
        print(f"SiT: Filtered {(~outlier_mask).sum()} outliers from {len(outlier_mask)} points")
    
    if len(x_sit_valid) > 0:
        a_sit = y_sit_valid / x_sit_valid
        # Filter out any remaining NaN/inf from the division
        a_sit = a_sit[np.isfinite(a_sit)]
        if len(a_sit) > 0:
            B_sit = np.max(a_sit)
            A_sit = np.min(a_sit)
        else:
            B_sit = A_sit = 1.0  # Fallback
    else:
        B_sit = A_sit = 1.0  # Fallback
    
    print(f"SiT stats: {len(x_sit_valid)} valid points, y range [{np.min(y_sit_valid) if len(y_sit_valid)>0 else 0:.4f}, {np.max(y_sit_valid) if len(y_sit_valid)>0 else 0:.4f}], B/A = {B_sit/A_sit if A_sit != 0 else float('inf'):.2f}")
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # DINO plot
    if len(x_dino_valid) > 0:
        t_dino = np.linspace(np.min(x_dino_valid), np.max(x_dino_valid), 100)
        upper_line_dino = B_dino * t_dino
        lower_line_dino = A_dino * t_dino
        
        ax1.scatter(x_dino_valid, y_dino_valid, color='blue', alpha=0.5, label='Data points')
        ax1.plot(t_dino, upper_line_dino, color='red', linestyle='--', label='Upper bound')
        ax1.plot(t_dino, lower_line_dino, color='green', linestyle='--', label='Lower bound')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('DINO Distance (squared)', fontsize=12)
        ax1.set_ylabel('Symmetrized KL Divergence', fontsize=12)
        ratio_str = f'{B_dino/A_dino:.2f}' if A_dino != 0 else 'inf'
        ax1.set_title(f'DINO Feature Space\nRatio B/A = {ratio_str}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # SiT plot
    if len(x_sit_valid) > 0:
        t_sit = np.linspace(np.min(x_sit_valid), np.max(x_sit_valid), 100)
        upper_line_sit = B_sit * t_sit
        lower_line_sit = A_sit * t_sit
        
        ax2.scatter(x_sit_valid, y_sit_valid, color='orange', alpha=0.5, label='Data points')
        ax2.plot(t_sit, upper_line_sit, color='red', linestyle='--', label='Upper bound')
        ax2.plot(t_sit, lower_line_sit, color='green', linestyle='--', label='Lower bound')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('SiT Distance (squared)', fontsize=12)
        ax2.set_ylabel('Symmetrized KL Divergence', fontsize=12)
        ratio_str = f'{B_sit/A_sit:.2f}' if A_sit != 0 else 'inf'
        ax2.set_title(f'SiT Feature Space\nRatio B/A = {ratio_str}', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")
    
    # Save embeddings to .npy files
    base_path = os.path.splitext(save_path)[0]
    np.save(f"{base_path}_dinov2.npy", emb_dino)
    np.save(f"{base_path}_sit.npy", emb_sit)
    print(f"Saved embeddings to {base_path}_dinov2.npy and {base_path}_sit.npy")
        
def apply_weak_augmentation(image, strength='subtle'):
    """
    Apply weak augmentations to an image to create a semantically similar pair.
    
    Args:
        image: Tensor of shape [C, H, W] in range [0, 255]
        strength: 'subtle', 'mild', or 'moderate' - controls augmentation strength
    
    Returns:
        Augmented image tensor
    """
    # Normalize to [0, 1] for transforms
    img = image / 255.0
    
    if strength == 'subtle':
        # Very small changes for pairs at x≈0
        angle = np.random.uniform(-1, 1)  # ±1 degree rotation
        brightness = np.random.uniform(0.98, 1.02)
        contrast = np.random.uniform(0.98, 1.02)
    elif strength == 'mild':
        angle = np.random.uniform(-3, 3)
        brightness = np.random.uniform(0.95, 1.05)
        contrast = np.random.uniform(0.95, 1.05)
    else:  # moderate
        angle = np.random.uniform(-5, 5)
        brightness = np.random.uniform(0.9, 1.1)
        contrast = np.random.uniform(0.9, 1.1)
    
    # Apply rotation
    img = TF.rotate(img, angle, fill=0.5)
    
    # Apply color jitter
    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)
    
    # Random small crop and resize
    if strength != 'subtle':
        h, w = img.shape[-2:]
        crop_ratio = 0.95 if strength == 'mild' else 0.9
        crop_size = int(min(h, w) * crop_ratio)
        img = TF.center_crop(img, crop_size)
        img = TF.resize(img, [h, w])
    
    # Denormalize back to [0, 255]
    img = torch.clamp(img * 255.0, 0, 255)
    
    return img


def slerp_features(u, v, t):
    """
    Spherical Linear Interpolation (SLERP) between two feature vectors on a hypersphere.
    
    Args:
        u, v: Feature tensors (already normalized to unit norm)
        t: Interpolation parameter in [0, 1]. t=0 returns u, t=1 returns v
    
    Returns:
        Interpolated feature vector on the hypersphere
    """
    # Compute the angle between vectors
    dot_product = torch.sum(u * v, dim=-1, keepdim=True)
    # Clamp to avoid numerical issues with arccos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    omega = torch.acos(dot_product)
    
    # Handle near-parallel vectors (omega ≈ 0) with linear interpolation
    # This avoids division by sin(omega) ≈ 0
    sin_omega = torch.sin(omega)
    near_parallel = sin_omega < 1e-6
    
    if near_parallel.any():
        # Use linear interpolation for near-parallel vectors
        result = (1 - t) * u + t * v
        # Re-normalize to stay on the sphere
        result = result / torch.linalg.norm(result, ord=2, dim=-1, keepdim=True)
    else:
        # Use SLERP formula
        coeff_u = torch.sin((1 - t) * omega) / sin_omega
        coeff_v = torch.sin(t * omega) / sin_omega
        result = coeff_u * u + coeff_v * v
    
    return result


def get_stratified_pairs(dataset, dino_model, feature_type, sit_model, vae, latents_stats, 
                        use_sdvae, sit_depth, use_projector, average_features, device,
                        n_samples=2000, n_bins=20, samples_per_bin=50, close_threshold=100.0,
                        pairs_file=None):
    """
    Generate stratified pairs by binning pairwise distances and filling bins uniformly.
    Uses SLERP interpolation to fill empty bins (typically at low distances).
    
    Args:
        n_samples: Number of images to sample for computing pairwise distances
        n_bins: Number of distance bins
        samples_per_bin: Target number of pairs per bin
        close_threshold: Distance threshold below which to use SLERP for empty bins
        pairs_file: Path to load/save pairs. If file exists, loads pairs. Otherwise creates and saves.
    
    Returns:
        List of (img1_idx, img2_idx, slerp_t) tuples. slerp_t is None for natural pairs.
    """
    # Try to load existing pairs if file is specified
    if pairs_file is not None and os.path.exists(pairs_file):
        print(f"Loading existing pairs from {pairs_file}...")
        pairs_data = np.load(pairs_file, allow_pickle=True)
        selected_pairs = pairs_data['pairs'].tolist()
        print(f"Loaded {len(selected_pairs)} pairs")
        return selected_pairs
    
    print(f"Computing stratified pairs from {n_samples} images...")
    
    # 1. Sample random images and extract features
    idx = np.random.choice(len(dataset), n_samples, replace=False)
    
    print("Extracting features for stratification...")
    all_features = []
    for i in tqdm(idx, desc="Feature extraction"):
        img = dataset[i][0].unsqueeze(0).to(device)
        with torch.no_grad():
            if feature_type == 'dino':
                feat = extract_dino_features(dino_model, "dinov2-vit-b", img, average_features=average_features)
            else:
                feat = extract_sit_features(sit_model, vae, img, sit_depth, latents_stats, 
                                           use_sdvae, use_projector, average_features=average_features)
            all_features.append(feat)
    
    all_features = torch.cat(all_features, dim=0)  # [N, D] or [N, num_patches, D]
    all_features = all_features / torch.linalg.norm(all_features, ord=2, dim=-1, keepdim=True)  # Normalize
    # Flatten spatial dimensions if needed for distance computation
    if len(all_features.shape) > 2:
        N = all_features.shape[0]
        num_patches = all_features.shape[1]  # Store before flattening
        all_features_flat = all_features.reshape(N, -1)
    else:
        all_features_flat = all_features
        num_patches = 1  # Averaged features: only 1 "patch"
    
    # 2. Compute pairwise distances (squared L2 normalized by dimension)
    print("Computing pairwise distances...")
    
    dists = torch.cdist(all_features_flat, all_features_flat, p=2).pow(2)
    # For unit-normalized vectors: ||u-v||^2 range is [0, 4]
    # For num_patches concatenated unit vectors: range is [0, 4*num_patches]
    # Divide by num_patches (mean across patches) then by 2 to get [0, 2]
    dists = dists / num_patches / 2
    
    # Get upper triangle (avoid duplicates and self-pairs)
    triu_indices = torch.triu_indices(n_samples, n_samples, offset=1)
    dists_flat = dists[triu_indices[0], triu_indices[1]].cpu()  # Move to CPU
    
    # Free VRAM from large distance matrix
    del dists
    torch.cuda.empty_cache()
    
    # 3. Define bins
    min_d, max_d = dists_flat.min().item(), dists_flat.max().item()
    bins = np.linspace(min_d, max_d, n_bins + 1)
    
    print(f"Distance range: [{min_d:.4f}, {max_d:.4f}]")
    
    selected_pairs = []
    
    # 4. Fill bins from natural data
    for i in range(n_bins):
        bin_min, bin_max = bins[i], bins[i+1]
        mask = (dists_flat >= bin_min) & (dists_flat < bin_max)
        bin_indices = torch.where(mask)[0]
        
        if len(bin_indices) > 0:
            # Natural pairs exist in this bin
            n_to_sample = min(samples_per_bin, len(bin_indices))
            chosen = bin_indices[torch.randperm(len(bin_indices))[:n_to_sample]]
            
            for pair_idx in chosen:
                i_idx = triu_indices[0][pair_idx].item()
                j_idx = triu_indices[1][pair_idx].item()
                selected_pairs.append((idx[i_idx], idx[j_idx], None))
            
            print(f"Bin {i+1}/{n_bins} [{bin_min:.4f}, {bin_max:.4f}]: {n_to_sample} natural pairs")
        else:
            # Empty bin - use SLERP if below threshold
            if bin_max < close_threshold:
                # Target distance is bin center
                target_dist = (bin_min + bin_max) / 2
                
                # Generate SLERP pairs by finding appropriate t value
                # We need to sample random image pairs and find t such that distance ≈ target_dist
                n_generated = 0
                attempts = 0
                max_attempts = samples_per_bin * 10
                last_t = None  # Track the last t value used
                
                while n_generated < samples_per_bin and attempts < max_attempts:
                    attempts += 1
                    # Sample random pair
                    i_idx, j_idx = np.random.choice(n_samples, 2, replace=False)
                    
                    # Get their features
                    feat_i = all_features[i_idx]
                    feat_j = all_features[j_idx]
                    
                    # Normalize
                    feat_i_norm = feat_i / torch.linalg.norm(feat_i, ord=2, dim=-1, keepdim=True)
                    feat_j_norm = feat_j / torch.linalg.norm(feat_j, ord=2, dim=-1, keepdim=True)
                    
                    # Compute their distance using same normalization as above
                    base_dist = torch.norm(feat_i_norm - feat_j_norm, p=2).item()**2 / num_patches / 2
                    
                    # Find t such that SLERP(feat_i, feat_j, t) has distance ≈ target_dist from feat_i
                    # For SLERP, the relationship is approximately: distance(t) ≈ t * base_dist for small t
                    # We need base_dist to be large enough that we can reach target_dist with reasonable t
                    if base_dist > target_dist:  # Base distance should be larger than target
                        # Compute t to achieve target distance
                        # Since distance scales roughly linearly with t for small t: d(t) ≈ t * base_dist
                        t = target_dist / base_dist
                        t = np.clip(t, 0.01, 0.5)  # Allow larger t values to reach target
                        
                        # Verify the actual distance achieved
                        feat_interp = slerp_features(feat_i_norm, feat_j_norm, t)
                        actual_dist = torch.norm(feat_i_norm - feat_interp, p=2).item()**2 / num_patches / 2
                        
                        # Accept if within reasonable tolerance of target
                        if 0.5 * target_dist <= actual_dist <= 2.0 * target_dist:
                            selected_pairs.append((idx[i_idx], idx[j_idx], t))
                            last_t = t  # Track this t value
                            n_generated += 1
                
                if n_generated > 0:
                    print(f"Bin {i+1}/{n_bins} [{bin_min:.4f}, {bin_max:.4f}]: {n_generated} SLERP pairs (t≈{last_t:.3f})")
                else:
                    print(f"Bin {i+1}/{n_bins} [{bin_min:.4f}, {bin_max:.4f}]: 0 SLERP pairs (failed to generate)")
            else:
                print(f"Bin {i+1}/{n_bins} [{bin_min:.4f}, {bin_max:.4f}]: EMPTY (skipped)")
    
    print(f"Total selected pairs: {len(selected_pairs)}")
    
    # Save pairs if file path is specified
    if pairs_file is not None:
        os.makedirs(os.path.dirname(pairs_file), exist_ok=True)
        np.savez(pairs_file, pairs=np.array(selected_pairs, dtype=object))
        print(f"Saved pairs to {pairs_file}")
    
    return selected_pairs


def euclidean_embedding_inequality(model, num_pairs, dataset, dino_model, sit_model, vae, 
                                   latents_stats, use_sdvae, sit_depth, feature_type, use_projector,
                                   device='cuda', mc_sample=1, num_steps=20, lamda=30000, average_features=True, gibbs=False,
                                   use_augmented_pairs=False, augmentation_mode='mixed', 
                                   use_stratified=False, n_bins=20, samples_per_bin=50, n_samples=20000,
                                   pairs_file=None):
    """
    Compute euclidean embedding inequality using both DINO and SiT features.
    Features can be spatially averaged or kept as spatial maps based on average_features flag.
    KL divergence is computed using features specified by feature_type.
    
    Args:
        use_augmented_pairs: If True, construct semantically similar pairs instead of random pairs
        augmentation_mode: 'weak' (weak augmentations + random), 'interpolation' (SLERP on features),
                          or 'mixed' (combination of both methods to sweep the full distance range)
        use_stratified: If True, use stratified sampling to ensure uniform coverage of distance bins
        n_bins: Number of bins for stratified sampling
        samples_per_bin: Target number of pairs per bin for stratified sampling
    """
    embeddings_dino = []
    embeddings_sit = []
    
    # Use stratified sampling if requested
    if use_stratified:
        pair_list = get_stratified_pairs(
            dataset, dino_model, feature_type, sit_model, vae, latents_stats,
            use_sdvae, sit_depth, use_projector, average_features, device,
            n_samples=n_samples, n_bins=n_bins, samples_per_bin=samples_per_bin,
            pairs_file=pairs_file
        )
        num_pairs = len(pair_list)
    else:
        pair_list = None
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
    
    if use_stratified:
        # Stratified sampling already computed pair_list
        pass
    elif use_augmented_pairs:
        # For augmented pairs, we need different strategies based on mode
        if augmentation_mode == 'weak':
            # Use weak augmentations (50%) + random pairs (50%)
            num_aug = num_pairs // 2
            num_random = num_pairs - num_aug
            dataset_subset = torch.utils.data.Subset(dataset, idx[:num_aug + num_random*2])
            augmentation_strengths = ['subtle', 'mild', 'moderate'] * (num_aug // 3 + 1)
        elif augmentation_mode == 'interpolation':
            # Use SLERP on features - sweeps the full distance range
            # Select random pairs, will compute interpolation in the loop
            dataset_subset = torch.utils.data.Subset(dataset, idx[:num_pairs*2])
            # Generate mixing coefficients to sweep from small to large distances
            # More points near 0 to populate the left side of the plot (small t values)
            alphas = np.concatenate([
                np.linspace(0.01, 0.2, num_pairs // 2),  # Small steps for close pairs
                np.linspace(0.2, 0.5, num_pairs // 4),   # Medium distances
                np.linspace(0.5, 1.0, num_pairs - 3*(num_pairs // 4))  # Full range
            ])
            np.random.shuffle(alphas)
        else:  # 'mixed'
            # Combine both methods for comprehensive coverage
            num_aug = num_pairs // 2
            num_interp = num_pairs - num_aug
            dataset_subset = torch.utils.data.Subset(dataset, idx[:num_aug + num_interp*2])
            augmentation_strengths = ['subtle', 'mild', 'moderate'] * (num_aug // 3 + 1)
            alphas = np.concatenate([
                np.linspace(0.01, 0.2, num_interp // 2),
                np.linspace(0.2, 0.5, num_interp // 4),
                np.linspace(0.5, 1.0, num_interp - 3*(num_interp // 4))
            ])
            np.random.shuffle(alphas)
    
    for i in tqdm(range(num_pairs)):
        # Step 1: Load images based on mode
        if use_stratified:
            # Use pre-computed stratified pairs
            idx1, idx2, slerp_t = pair_list[i]
            img1 = dataset[idx1][0].unsqueeze(0).to(device)
            img2 = dataset[idx2][0].unsqueeze(0).to(device)
        elif use_augmented_pairs and augmentation_mode == 'weak':
            if i < num_aug:
                # First half: weak augmentation
                img1 = dataset_subset[i][0].unsqueeze(0).to(device)
                img2 = apply_weak_augmentation(img1.squeeze(0), strength=augmentation_strengths[i]).unsqueeze(0)
            else:
                # Second half: random pairs
                idx_offset = num_aug
                random_idx = i - num_aug
                img1 = dataset_subset[idx_offset + random_idx*2][0].unsqueeze(0).to(device)
                img2 = dataset_subset[idx_offset + random_idx*2 + 1][0].unsqueeze(0).to(device)
        elif use_augmented_pairs and augmentation_mode == 'interpolation':
            # SLERP: load two images
            img1 = dataset_subset[i*2][0].unsqueeze(0).to(device)
            img2 = dataset_subset[i*2 + 1][0].unsqueeze(0).to(device)
        elif use_augmented_pairs and augmentation_mode == 'mixed':
            if i < num_aug:
                # First half: weak aug
                img1 = dataset_subset[i][0].unsqueeze(0).to(device)
                img2 = apply_weak_augmentation(img1.squeeze(0), strength=augmentation_strengths[i]).unsqueeze(0)
            else:
                # Second half: SLERP
                idx_offset = num_aug
                interp_idx = i - num_aug
                img1 = dataset_subset[idx_offset + interp_idx*2][0].unsqueeze(0).to(device)
                img2 = dataset_subset[idx_offset + interp_idx*2 + 1][0].unsqueeze(0).to(device)
        else:
            # Random pairs
            img1 = dataset_subset[i*2][0].unsqueeze(0).to(device)
            img2 = dataset_subset[i*2 + 1][0].unsqueeze(0).to(device)
        
        # Step 2: Extract features from both images
        with torch.no_grad():
            phi_1_dino = extract_dino_features(dino_model, "dinov2-vit-b", img1, average_features=average_features)
            phi_2_dino = extract_dino_features(dino_model, "dinov2-vit-b", img2, average_features=average_features)
            phi_1_sit = extract_sit_features(sit_model, vae, img1, sit_depth, latents_stats, use_sdvae, use_projector, average_features=average_features)
            phi_2_sit = extract_sit_features(sit_model, vae, img2, sit_depth, latents_stats, use_sdvae, use_projector, average_features=average_features)
        
        # Step 3: Apply SLERP if in interpolation mode or stratified with SLERP
        should_apply_slerp = False
        t = None
        
        if use_stratified and slerp_t is not None:
            # Stratified mode with SLERP pair
            should_apply_slerp = True
            t = slerp_t
        elif use_augmented_pairs and augmentation_mode in ['interpolation', 'mixed']:
            if augmentation_mode == 'interpolation' or (augmentation_mode == 'mixed' and i >= num_aug):
                should_apply_slerp = True
                t = alphas[i] if augmentation_mode == 'interpolation' else alphas[i - num_aug]
        
        phi_1_dino = phi_1_dino / torch.linalg.norm(phi_1_dino, ord=2, dim=-1, keepdim=True)
        phi_2_dino = phi_2_dino / torch.linalg.norm(phi_2_dino, ord=2, dim=-1, keepdim=True)
        phi_1_sit = phi_1_sit / torch.linalg.norm(phi_1_sit, ord=2, dim=-1, keepdim=True)
        phi_2_sit = phi_2_sit / torch.linalg.norm(phi_2_sit, ord=2, dim=-1, keepdim=True)
        
        if should_apply_slerp:
            # Normalize and interpolate
            # Keep phi_1 as reference, interpolate phi_2
            phi_1_dino, phi_2_dino = phi_1_dino, slerp_features(phi_1_dino, phi_2_dino, t)
            phi_1_sit, phi_2_sit = phi_1_sit, slerp_features(phi_1_sit, phi_2_sit, t)
        
        # Choose features for conditioning based on feature_type
        if feature_type == 'dino':
            phi_1_cond = phi_1_dino
            phi_2_cond = phi_2_dino
        else:  # 'sit'
            phi_1_cond = phi_1_sit
            phi_2_cond = phi_2_sit
        
        # Compute KL divergence using selected features for conditioning
        # Add batch dimension if features are averaged, otherwise features already have [B, num_patches, D] shape
        if average_features:
            dist_kl = symetrized_kl(model, phi_1_cond.unsqueeze(1), phi_2_cond.unsqueeze(1), 
                                   mc_sample=mc_sample, num_steps=num_steps, device=device, lamda=lamda,
                                   feature_type=feature_type, use_projector=use_projector, gibbs=gibbs, average_features=average_features)
        else:
            dist_kl = symetrized_kl(model, phi_1_cond, phi_2_cond, 
                                   mc_sample=mc_sample, num_steps=num_steps, device=device, lamda=lamda,
                                   feature_type=feature_type, use_projector=use_projector, average_features=average_features)
        
        # Compute distances in both feature spaces
        
        dist_dino = torch.norm(phi_1_dino - phi_2_dino, p=2).item()**2 / phi_1_dino.shape[-2] / 2
        dist_sit = torch.norm(phi_1_sit - phi_2_sit, p=2).item()**2 / phi_1_sit.shape[-2] / 2

        embeddings_dino.append((dist_dino, dist_kl.item()))
        embeddings_sit.append((dist_sit, dist_kl.item()))
    
    return embeddings_dino, embeddings_sit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Euclidean Embedding Inequality Analysis')
    # Data arguments
    parser.add_argument('--data-dir', type=str, default="/datasets_local/nsereyjo/ImageNet")
    parser.add_argument('--model-path', type=str, default="pretrained/sit-ldm-e2e-vavae/checkpoints/4000000.pt")
    
    # Analysis arguments
    parser.add_argument('--num-pairs', type=int, default=1000)
    parser.add_argument('--mc-sample', type=int, default=20)
    parser.add_argument('--num-steps', type=int, default=250)
    parser.add_argument('--lamda', type=float, default=30000)
    parser.add_argument('--output-path', type=str, default="log/results_debug/embedding_inequality_log.png")
    parser.add_argument('--gibbs', action='store_true',
                       help='Use Gibbs scaling in score computation')
    
    # Monte Carlo variance analysis
    parser.add_argument('--analyze-mc-variance', action='store_true',
                       help='Analyze how variance evolves with number of MC samples instead of computing full inequality')
    parser.add_argument('--max-mc-samples', type=int, default=100,
                       help='Maximum number of MC samples to test (for variance analysis)')
    parser.add_argument('--num-repeats', type=int, default=10,
                       help='Number of times to repeat each k value (for variance analysis)')
    parser.add_argument('--num-test-pairs', type=int, default=5,
                       help='Number of image pairs to test (for variance analysis)')
    
    # Feature extraction arguments
    parser.add_argument('--feature-type', type=str, default='dino', choices=['dino', 'sit'],
                       help='Which features to use for conditioning (dino or sit)')
    parser.add_argument('--sit-depth', type=int, default=8,
                       help='Which SiT layer to extract features from')
    parser.add_argument('--use-projector', action='store_true',
                       help='Use projected SiT features (default: use non-projected features)')
    parser.add_argument('--average-features', action='store_true', default=False,
                       help='Average features spatially and use MeanFeatAlignment potential. '
                            'If False, uses full spatial feature maps with RepaPotential.')
    
    # Augmented pairs arguments
    parser.add_argument('--use-augmented-pairs', action='store_true',
                       help='Construct semantically similar pairs using augmentations/interpolation '
                            'instead of random pairs to reveal the global linear trend')
    parser.add_argument('--augmentation-mode', type=str, default='mixed',
                       choices=['weak', 'interpolation', 'mixed'],
                       help='How to construct augmented pairs: '
                            '"weak" uses weak augmentations (50%%) + random pairs (50%%), '
                            '"interpolation" uses SLERP on features (sweeps full range on hypersphere), '
                            '"mixed" combines weak augmentations (50%%) + SLERP (50%%)')
    
    # Stratified sampling arguments
    parser.add_argument('--use-stratified', action='store_true',
                       help='Use stratified sampling to ensure uniform coverage of distance bins. '
                            'Bins with no natural pairs are filled using SLERP interpolation.')
    parser.add_argument('--n-bins', type=int, default=20,
                       help='Number of distance bins for stratified sampling')
    parser.add_argument('--samples-per-bin', type=int, default=50,
                       help='Target number of pairs per bin for stratified sampling')
    parser.add_argument('--n-samples', type=int, default=20000,
                       help='Number of images to sample for computing pairwise distances in stratified sampling')
    parser.add_argument('--pairs-file', type=str, default=None,
                       help='Path to load/save pairs for consistent comparison. If exists, loads pairs; '
                            'otherwise creates and saves them. Format: log/pairs/stratified_bins{n_bins}_samples{n_samples}.npz')
    
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    dataset = CustomINH5Dataset(args.data_dir)

    gc.collect()

    # Load SiT model and VAE
    # Config parsing is handled internally by load_sit_and_vae
    # If no config exists, defaults to Stable Diffusion VAE with fixed hyperparameters
    print(f"Loading SiT model from {args.model_path}")
    model, vae, latents_scale, latents_bias, sit_ckpt, _, _ = load_sit_and_vae(
        checkpoint_path=args.model_path,
        device=device,
    )
    
    # Determine if we're using SD-VAE by checking the VAE type
    use_sdvae = isinstance(vae, type(vae)) and hasattr(vae, 'config')  # SD-VAE from diffusers has config attribute
    
    # Create latents_stats dict for compatibility with extract_sit_features
    latents_stats = {
        'scale': latents_scale,
        'bias': latents_bias
    }
    
    # Always load both DINO model and VAE for SiT features
    print("Loading DINO model...")
    dino_model = load_encoders("dinov2-vit-b", device, 256)[0][0]
    dino_model.eval()
    
    print("Loading SiT model for feature extraction...")
    sit_feature_model = model
    
    if args.analyze_mc_variance:
        # Run MC variance analysis on a few test pairs
        print(f"\n=== Monte Carlo Variance Analysis ===")
        print(f"Testing {args.num_test_pairs} image pairs")
        print(f"Pair construction: {'Augmented (' + args.augmentation_mode + ')' if args.use_augmented_pairs else 'Random'}")
        print(f"Using {args.feature_type.upper()} features for conditioning")
        print(f"SiT features: {'projected' if args.use_projector else 'non-projected'}")
        print(f"Features: {'averaged (MeanFeatAlignment)' if args.average_features else 'spatial maps (RepaPotential)'}")
        
        # Select random image pairs
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        test_indices = idx[:args.num_test_pairs * 2]
        
        all_mc_samples = []
        all_means = []
        all_stds = []
        
        for pair_idx in range(args.num_test_pairs):
            # Load images based on pair construction mode
            if args.use_augmented_pairs and args.augmentation_mode != 'weak':
                # For SLERP: load two images
                img1 = dataset[test_indices[pair_idx*2]][0].unsqueeze(0).to(device)
                img2 = dataset[test_indices[pair_idx*2 + 1]][0].unsqueeze(0).to(device)
            else:
                # For weak aug or random: load one or two images
                img1 = dataset[test_indices[pair_idx]][0].unsqueeze(0).to(device)
                img2 = None
            
            # Extract features from images
            with torch.no_grad():
                if args.feature_type == 'dino':
                    phi_1 = extract_dino_features(dino_model, "dinov2-vit-b", img1, average_features=args.average_features)
                    if img2 is not None:
                        phi_2 = extract_dino_features(dino_model, "dinov2-vit-b", img2, average_features=args.average_features)
                else:
                    phi_1 = extract_sit_features(sit_feature_model, vae, img1, args.sit_depth, latents_stats, use_sdvae, args.use_projector, average_features=args.average_features)
                    if img2 is not None:
                        phi_2 = extract_sit_features(sit_feature_model, vae, img2, args.sit_depth, latents_stats, use_sdvae, args.use_projector, average_features=args.average_features)
            
            # Apply augmentation strategy
            if args.use_augmented_pairs:
                if args.augmentation_mode == 'weak' or (args.augmentation_mode == 'mixed' and pair_idx < args.num_test_pairs // 2):
                    # Weak augmentation in feature space (re-extract from augmented image)
                    img1_aug = apply_weak_augmentation(img1.squeeze(0), strength='mild').unsqueeze(0)
                    with torch.no_grad():
                        if args.feature_type == 'dino':
                            phi_2 = extract_dino_features(dino_model, "dinov2-vit-b", img1_aug, average_features=args.average_features)
                        else:
                            phi_2 = extract_sit_features(sit_feature_model, vae, img1_aug, args.sit_depth, latents_stats, use_sdvae, args.use_projector, average_features=args.average_features)
                else:
                    # SLERP interpolation
                    t = np.random.uniform(0.05, 0.2)
                    phi_1 = phi_1 / torch.linalg.norm(phi_1, ord=2, dim=-1, keepdim=True)
                    phi_2 = phi_2 / torch.linalg.norm(phi_2, ord=2, dim=-1, keepdim=True)
                    phi_2 = slerp_features(phi_1, phi_2, t)
            
            print(f"\nAnalyzing pair {pair_idx + 1}/{args.num_test_pairs}...")
            # Handle dimension based on averaging
            if args.average_features:
                phi_1_input = phi_1.unsqueeze(1)
                phi_2_input = phi_2.unsqueeze(1)
            else:
                phi_1_input = phi_1
                phi_2_input = phi_2
            mc_samples, means, stds = analyze_mc_variance(
                model, phi_1_input, phi_2_input,
                max_mc_samples=args.max_mc_samples,
                num_repeats=args.num_repeats,
                num_steps=args.num_steps,
                device=device,
                lamda=args.lamda,
                feature_type=args.feature_type,
                use_projector=args.use_projector,
                average_features=args.average_features
            )
            
            all_mc_samples.append(mc_samples)
            all_means.append(means)
            all_stds.append(stds)
        
        # Average results across pairs
        avg_means = np.mean(all_means, axis=0)
        avg_stds = np.mean(all_stds, axis=0)
        
        # Visualize averaged results
        output_dir = os.path.dirname(args.output_path)
        variance_path = os.path.join(output_dir, "mc_variance_analysis.png")
        os.makedirs(output_dir, exist_ok=True)
        visualize_mc_variance(mc_samples, avg_means, avg_stds, save_path=variance_path)
        
    else:
        # Original inequality computation
        print(f"Computing euclidean embedding inequality for {args.num_pairs} pairs...")
        if args.use_stratified:
            print(f"Pair construction: Stratified sampling with {args.n_bins} bins, {args.samples_per_bin} samples/bin, from {args.n_samples} images")
        elif args.use_augmented_pairs:
            print(f"Pair construction: Augmented ({args.augmentation_mode})")
        else:
            print(f"Pair construction: Random")
        print(f"Using {args.feature_type.upper()} features for conditioning")
        print(f"SiT features: {'projected' if args.use_projector else 'non-projected'}")
        print(f"Features: {'averaged (MeanFeatAlignment)' if args.average_features else 'spatial maps (RepaPotential)'}")
        embeddings_dino, embeddings_sit = euclidean_embedding_inequality(
            model, args.num_pairs, dataset, dino_model,
            sit_model=sit_feature_model, vae=vae, latents_stats=latents_stats,
            use_sdvae=use_sdvae, sit_depth=args.sit_depth, feature_type=args.feature_type,
            use_projector=args.use_projector,
            device=device, mc_sample=args.mc_sample, num_steps=args.num_steps, lamda=args.lamda,
            average_features=args.average_features, gibbs=args.gibbs,
            use_augmented_pairs=args.use_augmented_pairs, augmentation_mode=args.augmentation_mode,
            use_stratified=args.use_stratified, n_bins=args.n_bins, samples_per_bin=args.samples_per_bin,
            n_samples=args.n_samples, pairs_file=args.pairs_file
        )
        
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        visu_paper(embeddings_dino, save_path=args.output_path)
        print(f"\nAnalysis complete! Results saved to {args.output_path}")
    
    

