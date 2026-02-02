import math
import warnings
import os
import json

import numpy as np
from PIL import Image
import timm
import torch

from models import mocov3_vit
from models.sit import SiT_models
from models.autoencoder import vae_models

try:
    from diffusers import AutoencoderKL
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


def _compute_latent_dimensions(vae_type, resolution=256):
    """Helper function to compute latent dimensions from VAE type and resolution."""
    if vae_type == "f8d4":
        return resolution // 8, 4
    elif vae_type == "f16d32":
        return resolution // 16, 32
    elif vae_type == "h16d4":
        return (resolution, resolution // 16), 4
    elif vae_type == "h2p4d8":
        return (resolution // 4, resolution // 8), 8
    elif vae_type == "h1p4d4":
        return (resolution // 4, resolution // 4), 4
    else:
        raise NotImplementedError(f"Unsupported VAE type: {vae_type}")


def load_sit_and_vae(
    checkpoint_path,
    device,
    config=None,
):
    """
    Load a SiT-XL/2 model and VAE with automatic config parsing.
    
    This function automatically loads configuration from args.json in the checkpoint
    directory. If no config exists, it uses Stable Diffusion VAE with fixed defaults.
    
    Args:
        checkpoint_path: Path to the SiT checkpoint file
        device: Device to load the models on
        config: Optional config object/dict. If provided, will use its parameters.
                If None, will try to load from args.json in checkpoint directory.
    
    Returns:
        tuple: (model, vae, latents_scale, latents_bias, checkpoint_dict, latent_size, config)
            - model: Loaded SiT model
            - vae: Loaded VAE model
            - latents_scale: Scale for latent normalization
            - latents_bias: Bias for latent normalization
            - checkpoint_dict: Full checkpoint for additional information
            - latent_size: Latent spatial dimensions
            - config: Config object (created if None was provided)
    """
    
    # Create config if None is provided
    config_was_none = config is None
    
    # Extract parameters from config if available
    if config is not None:
        model_name = config.model
        vae_type = config.vae
        resolution = getattr(config, 'resolution', 256)
        num_classes = getattr(config, 'num_classes', 1000)
        encoder_depth = getattr(config, 'encoder_depth', 8)
        
        # Parse encoder dimensions from config
        if hasattr(config, 'enc_type') and config.enc_type != 'None':
            # Load encoder temporarily to get dimensions
            encoders, _, _ = load_encoders(config.enc_type, "cpu", resolution)
            z_dims = [encoder.embed_dim for encoder in encoders]
            del encoders
        else:
            z_dims = [768]  # Default
        
        class_dropout_prob = getattr(config, 'cfg_prob', 0.1)
        bn_momentum = getattr(config, 'bn_momentum', 0.01)
        fused_attn = getattr(config, 'fused_attn', True)
        qk_norm = getattr(config, 'qk_norm', False)
        
        # Get VAE checkpoint path from config
        vae_ckpt_path = getattr(config, 'vae_ckpt', None)
        use_sdvae = getattr(config, 'use_sdvae', False)
        strict = True
        fix_shape_mismatch = False
    else:
        # No config available - use SD-VAE with fixed defaults
        model_name = "SiT-XL/2"
        vae_type = "f8d4"
        resolution = 256
        num_classes = 1000
        encoder_depth = 8
        z_dims = [768]
        class_dropout_prob = 0.1
        bn_momentum = 0.01
        fused_attn = False
        qk_norm = False
        vae_ckpt_path = None
        use_sdvae = True
        strict = False
        fix_shape_mismatch = True
    
    # Compute latent dimensions
    latent_size, in_channels = _compute_latent_dimensions(vae_type, resolution)
    block_kwargs = {"fused_attn": fused_attn, "qk_norm": qk_norm}
    
    # Initialize SiT model
    model = SiT_models[model_name](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=class_dropout_prob,
        z_dims=z_dims,
        encoder_depth=encoder_depth,
        bn_momentum=bn_momentum,
        **block_kwargs,
    ).to(device)
    
    # Load SiT checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract state dict from checkpoint (handle different formats)
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        # Assume the checkpoint is the state dict itself
        state_dict = checkpoint
        

    # Handle final layer shape mismatches if requested
    if fix_shape_mismatch:
        if "final_layer.linear.bias" in state_dict:
            expected_size = model.final_layer.linear.bias.shape[0]
            actual_size = state_dict["final_layer.linear.bias"].shape[0]
            print("actual_size:", actual_size, "expected_size:", expected_size)
            
            if actual_size != expected_size:
                
                def signal_indices(p, C):
                    idx = []
                    for i in range(p):
                        for j in range(p):
                            for c in range(C):          # signal only
                                linear_idx = (i * p + j) * (2 * C) + c
                                idx.append(linear_idx)
                    return torch.tensor(idx, dtype=torch.long)
                
                idx = signal_indices(2,4)
                
                state_dict["final_layer.linear.weight"] = state_dict["final_layer.linear.weight"][idx, :]
                state_dict["final_layer.linear.bias"] = state_dict["final_layer.linear.bias"][idx]
    
    # Load SiT state dict
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    
    # Load VAE
    if use_sdvae:
        # Use Stable Diffusion VAE from diffusers
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library required for SD-VAE. Install with: pip install diffusers")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        vae.eval()
        # SD-VAE uses fixed scale/bias
        latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
        latents_bias = torch.zeros(1, 4, 1, 1).to(device)
    else:
        # Use custom VAE from vae_models
        vae = vae_models[vae_type]().to(device)
        
        # Check if VAE is in the SiT checkpoint (REPA-E training)
        if "vae" in checkpoint:
            # REPA-E checkpoints: VAE is in the checkpoint
            vae_state_dict = checkpoint['vae']
            vae.load_state_dict(vae_state_dict)
            
            # Extract latent stats from SiT's batch norm
            latents_scale = state_dict["bn.running_var"].rsqrt().view(1, in_channels, 1, 1).to(device)
            latents_bias = state_dict["bn.running_mean"].view(1, in_channels, 1, 1).to(device)
        else:
            # LDM-only training: Load VAE from separate checkpoint
            if vae_ckpt_path is None:
                vae_ckpt_path = f"pretrained/{vae_type}/{vae_type}.pt"
            
            vae_checkpoint = torch.load(vae_ckpt_path, map_location=device)
            
            # Handle different VAE checkpoint formats
            if 'ema' in vae_checkpoint:
                vae.load_state_dict(vae_checkpoint['ema'])
            elif 'model' in vae_checkpoint:
                vae.load_state_dict(vae_checkpoint['model'])
            else:
                vae.load_state_dict(vae_checkpoint)
            
            # Load latent stats from separate file
            stats_path = vae_ckpt_path.replace('.pt', '-latents-stats.pt')
            latents_stats = torch.load(stats_path, map_location=device)
            latents_scale = latents_stats["latents_scale"].to(device)
            latents_bias = latents_stats["latents_bias"].to(device)
        
        vae.eval()
    
    # Create config object if it was None
    if config_was_none:
        from dictdot import dictdot
        config = dictdot({
            'model': model_name,
            'vae': vae_type,
            'resolution': resolution,
            'num_classes': num_classes,
            'encoder_depth': encoder_depth,
            'enc_type': 'None',
            'cfg_prob': class_dropout_prob,
            'bn_momentum': bn_momentum,
            'fused_attn': fused_attn,
            'qk_norm': qk_norm,
            'vae_ckpt': vae_ckpt_path,
            'use_sdvae': use_sdvae,
        })
    
    return model, vae, latents_scale, latents_bias, checkpoint, latent_size, config


def fix_mocov3_state_dict(state_dict):
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder."):]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            if 'head' not in new_k and new_k.split('.')[0] != 'fc':
                state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if 'pos_embed' in state_dict.keys():
        state_dict['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
            state_dict['pos_embed'], [16, 16],
        )
    return state_dict

@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    assert (resolution == 256) or (resolution == 512)
    
    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        # Currently, we only support 512x512 experiments with DINOv2 encoders.
        if resolution == 512:
            if encoder_type != 'dinov2':
                raise NotImplementedError(
                    "Currently, we only support 512x512 experiments with DINOv2 encoders."
                    )

        architectures.append(architecture)
        encoder_types.append(encoder_type)
        if encoder_type == 'mocov3':
            if architecture == 'vit':
                if model_config == 's':
                    encoder = mocov3_vit.vit_small()
                elif model_config == 'b':
                    encoder = mocov3_vit.vit_base()
                elif model_config == 'l':
                    encoder = mocov3_vit.vit_large()
                ckpt = torch.load(f'./ckpts/mocov3_vit{model_config}.pth')
                state_dict = fix_mocov3_state_dict(ckpt['state_dict'])
                del encoder.head
                encoder.load_state_dict(state_dict, strict=True)
                encoder.head = torch.nn.Identity()
            elif architecture == 'resnet':
                raise NotImplementedError()
 
            encoder = encoder.to(device)
            encoder.eval()

        elif 'dinov2' in encoder_type:
            import timm
            if 'reg' in encoder_type:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2:qasfb-patch-3', f'dinov2_vit{model_config}14')
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()
        
        elif 'dinov1' == encoder_type:
            import timm
            from models import dinov1
            encoder = dinov1.vit_base()
            ckpt =  torch.load(f'./ckpts/dinov1_vit{model_config}.pth') 
            if 'pos_embed' in ckpt.keys():
                ckpt['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    ckpt['pos_embed'], [16, 16],
                )
            del encoder.head
            encoder.head = torch.nn.Identity()
            encoder.load_state_dict(ckpt, strict=True)
            encoder = encoder.to(device)
            encoder.forward_features = encoder.forward
            encoder.eval()

        elif encoder_type == 'clip':
            import clip
            from models.clip_vit import UpdatedVisionTransformer
            import os
            cache_root = os.path.join(os.environ["TORCH_HOME"], "clip")
            encoder_ = clip.load(f"ViT-{model_config}/14", device='cpu', download_root=cache_root)[0].visual
            encoder = UpdatedVisionTransformer(encoder_).to(device)
             #.to(device)
            encoder.embed_dim = encoder.model.transformer.width
            encoder.forward_features = encoder.forward
            encoder.eval()
        
        elif encoder_type == 'mae':
            from models.mae_vit import vit_large_patch16
            import timm
            kwargs = dict(img_size=256)
            encoder = vit_large_patch16(**kwargs).to(device)
            with open(f"ckpts/mae_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f)
            if 'pos_embed' in state_dict["model"].keys():
                state_dict["model"]['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    state_dict["model"]['pos_embed'], [16, 16],
                )
            encoder.load_state_dict(state_dict["model"])

            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [16, 16],
            )

        elif encoder_type == 'jepa':
            from models.jepa import vit_huge
            kwargs = dict(img_size=[224, 224], patch_size=14)
            encoder = vit_huge(**kwargs).to(device)
            with open(f"ckpts/ijepa_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f, map_location=device)
            new_state_dict = dict()
            for key, value in state_dict['encoder'].items():
                new_state_dict[key[7:]] = value
            encoder.load_state_dict(new_state_dict)
            encoder.forward_features = encoder.forward

        encoders.append(encoder)
    
    return encoders, encoder_types, architectures


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def center_crop_arr(image_arr, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    pil_image = Image.fromarray(image_arr)
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def preprocess_imgs_vae(imgs):
    # imgs: (B, C, H, W) -> (B, C, H, W), [0, 255] uint8 -> [-1, 1] float32
    return imgs.float() / 127.5 - 1.


def count_trainable_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def normalize_latents(latents, latents_scale, latents_bias):
    return (latents - latents_bias) * latents_scale


def denormalize_latents(latents, latents_scale, latents_bias):
    return latents / latents_scale + latents_bias


def extract_sit_features(model, vae, images, depth, latents_scale, latents_bias, use_sdvae=False, use_projector=False, average_features=False, time=0):
    """Extract SiT internal activations and optionally average spatially.
    
    Args:
        model: SiT model
        vae: VAE model (either custom or SD-VAE)
        images: Raw images tensor [B, C, H, W] in range [0, 255]
        depth: Which layer depth to extract features from
        latents_scale: Scale for latent normalization
        latents_bias: Bias for latent normalization
        use_sdvae: Whether using Stable Diffusion VAE (default: False)
        use_projector: Whether to use projected features (default: False)
        average_features: Whether to average features spatially (default: False)
    
    Returns:
        feats: Features [B, feature_dim] if average_features=True, else [B, num_patches, feature_dim]
    """
    with torch.no_grad():
        images_norm = images.float() / 127.5 - 1.0
        
        if use_sdvae:
            posterior = vae.encode(images_norm).latent_dist
            z = posterior.sample()
        else:
            posterior = vae.encode(images_norm)
            z = posterior.sample()
        
        z = normalize_latents(z, latents_scale, latents_bias)
        
        B = z.shape[0]
        device = z.device
        t = time * torch.ones(B, device=device)
        y = torch.full((B,), 1000, dtype=torch.long, device=device)
        
        feats = model.forward_feats(z, t, y, depth=depth, use_projector=use_projector)
        
        # When use_projector=True, feats is a list; when False, it's a tensor
        if use_projector:
            feats = feats[0]  # Take first projector output
        
        if average_features:
            feats = feats.mean(dim=1)  # Average spatially: [B, num_patches, D] -> [B, D]
            feats = feats.unsqueeze(1)  # Add spatial dim: [B, D] -> [B, 1, D]
        
    return feats

def compute_mask_from_pca(features_batch):
    """
    Computes a binary foreground mask from feature maps using PCA.
    Handles a batch of feature maps.
    """
    # Ensure input is a numpy array on CPU
    if isinstance(features_batch, torch.Tensor):
        features_batch = features_batch.cpu().numpy()

    # If the input is a single feature map, add a batch dimension
    if features_batch.ndim == 2:
        features_batch = np.expand_dims(features_batch, axis=0)

    B, L, D = features_batch.shape
    H = W = int(np.sqrt(L))
    n_components = 1  # Always use the first principal component for the mask

    batch_masks = []
    for i in range(B):
        feature_map = features_batch[i]  # Shape: (L, D)

        # Compute PCA
        feature_map_centered = feature_map - feature_map.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(feature_map_centered, full_matrices=False)
        
        # Project onto the first principal component
        pca_component = np.dot(feature_map_centered, Vt.T[:, :n_components]).flatten()

        # Normalize to [0, 1]
        pca_normalized = (pca_component - pca_component.min()) / (pca_component.max() - pca_component.min() + 1e-8)

        # Threshold at the median to create a binary mask
        threshold = 0.5
        binary_mask = (pca_normalized > threshold).astype(int)

        # --- Foreground/Background Heuristic ---
        # Check corner pixels of the 2D mask to guess if we selected the background
        mask_2d = binary_mask.reshape(H, W)
        corners = [mask_2d[0, 0], mask_2d[0, -1], mask_2d[-1, 0], mask_2d[-1, -1]]
        # If more than half the corners are selected, invert the mask
        if sum(corners) >= 2:
            binary_mask = 1 - binary_mask

        # --- Largest Connected Component ---
        # Keep only the largest connected component to remove noise
        from skimage.measure import label, regionprops
        labeled_mask = label(binary_mask.reshape(H, W))
        if labeled_mask.max() > 0:  # Check if there are any components
            regions = regionprops(labeled_mask)
            largest_region = max(regions, key=lambda r: r.area)
            # Create a new mask with only the largest component
            binary_mask = (labeled_mask == largest_region.label).flatten().astype(int)

        batch_masks.append(binary_mask)

    # Stack masks into a single numpy array of shape (B, L)
    final_masks = np.stack(batch_masks, axis=0)
    return torch.from_numpy(final_masks).float()