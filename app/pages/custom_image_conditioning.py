import streamlit as st
import torch
from dictdot import dictdot
import numpy as np
import sys
import os
from PIL import Image
from pathlib import Path
import json
from datetime import datetime

# Add the root REPA folder to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from potential import RepaPotential, TransportPotential, FreeEnergy, MultiPotential, MeanFeatAlignment

from app.utils import generate_image, display_image, load_model, compute_features, load_dino, display_interactive_feature_map, get_constant_noise


def center_crop_resize(img, target_size=256):
    """Center crop and resize image to target size"""
    width, height = img.size
    
    # Calculate center crop dimensions
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    
    # Crop to square
    img_cropped = img.crop((left, top, right, bottom))
    
    # Resize to target size
    img_resized = img_cropped.resize((target_size, target_size), Image.LANCZOS)
    
    return img_resized


def preprocess_image_for_dino(img, device):
    """Preprocess image for DINO feature extraction"""
    from torchvision import transforms
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        transforms.Resize((224, 224)),
    ])
    
    return transform(img).unsqueeze(0).to(device)


def load_images_from_folder(folder_path):
    """Load all images from a folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    folder = Path(folder_path)
    
    if not folder.exists():
        return []
    
    image_files = [f for f in folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    return sorted(image_files)



def save_generation_result(img_gen, img_1, img_2, metadata, output_folder, 
                           mask_1=None, mask_2=None, features_1=None, features_2=None, n_components=3,
                           img_3=None, mask_3=None, features_3=None):
    """Save generated image, input images, and metadata in a timestamped subfolder (supports 3 images)
    Args:
        img_gen: Generated image
        img_1, img_2, img_3: Input images
        metadata: Generation metadata
        output_folder: Base output folder
        mask_1, mask_2, mask_3: Optional masks for each image
        features_1, features_2, features_3: Optional feature maps for masked PCA visualization
        n_components: Number of PCA components for visualization
    """
    from app.utils import compute_pca_visualization
    output_path = Path(output_folder)
    # Generate timestamp-based subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = output_path / timestamp
    result_folder.mkdir(parents=True, exist_ok=True)
    # Save generated image
    gen_path = result_folder / "generated.png"
    Image.fromarray(img_gen).save(gen_path)
    # Save input images
    img_1_path = result_folder / "input1.png"
    img_2_path = result_folder / "input2.png"
    img_3_path = result_folder / "input3.png"
    if isinstance(img_1, np.ndarray):
        Image.fromarray(img_1).save(img_1_path)
    else:
        img_1.save(img_1_path)
    if isinstance(img_2, np.ndarray):
        Image.fromarray(img_2).save(img_2_path)
    else:
        img_2.save(img_2_path)
    if img_3 is not None:
        if isinstance(img_3, np.ndarray):
            Image.fromarray(img_3).save(img_3_path)
        else:
            img_3.save(img_3_path)
    # Save masks and masked PCA visualizations if provided
    if mask_1 is not None and features_1 is not None:
        mask_1_np = mask_1.squeeze(0).cpu().numpy() if torch.is_tensor(mask_1) else mask_1.squeeze(0)
        L = mask_1_np.shape[0]
        H = W = int(np.sqrt(L))
        mask_img = (mask_1_np.reshape(H, W) * 255).astype(np.uint8)
        mask_img_pil = Image.fromarray(mask_img, mode='L')
        mask_img_pil = mask_img_pil.resize((256, 256), Image.NEAREST)
        mask_img_pil.save(result_folder / "mask1.png")
        features_1_np = features_1.squeeze(0).cpu().numpy() if torch.is_tensor(features_1) else features_1.squeeze(0)
        feature_map_rgb, H, W = compute_pca_visualization(features_1_np, n_components)
        mask_3d = mask_1_np.reshape(H, W, 1).repeat(3, axis=2)
        masked_pca = (feature_map_rgb * mask_3d + 255 * (1 - mask_3d)).astype(np.uint8)
        masked_pca_pil = Image.fromarray(masked_pca)
        masked_pca_pil = masked_pca_pil.resize((256, 256), Image.NEAREST)
        masked_pca_pil.save(result_folder / "masked_pca1.png")
    if mask_2 is not None and features_2 is not None:
        mask_2_np = mask_2.squeeze(0).cpu().numpy() if torch.is_tensor(mask_2) else mask_2.squeeze(0)
        L = mask_2_np.shape[0]
        H = W = int(np.sqrt(L))
        mask_img = (mask_2_np.reshape(H, W) * 255).astype(np.uint8)
        mask_img_pil = Image.fromarray(mask_img, mode='L')
        mask_img_pil = mask_img_pil.resize((256, 256), Image.NEAREST)
        mask_img_pil.save(result_folder / "mask2.png")
        features_2_np = features_2.squeeze(0).cpu().numpy() if torch.is_tensor(features_2) else features_2.squeeze(0)
        feature_map_rgb, H, W = compute_pca_visualization(features_2_np, n_components)
        mask_3d = mask_2_np.reshape(H, W, 1).repeat(3, axis=2)
        masked_pca = (feature_map_rgb * mask_3d + 255 * (1 - mask_3d)).astype(np.uint8)
        masked_pca_pil = Image.fromarray(masked_pca)
        masked_pca_pil = masked_pca_pil.resize((256, 256), Image.NEAREST)
        masked_pca_pil.save(result_folder / "masked_pca2.png")
    if mask_3 is not None and features_3 is not None:
        mask_3_np = mask_3.squeeze(0).cpu().numpy() if torch.is_tensor(mask_3) else mask_3.squeeze(0)
        L = mask_3_np.shape[0]
        H = W = int(np.sqrt(L))
        mask_img = (mask_3_np.reshape(H, W) * 255).astype(np.uint8)
        mask_img_pil = Image.fromarray(mask_img, mode='L')
        mask_img_pil = mask_img_pil.resize((256, 256), Image.NEAREST)
        mask_img_pil.save(result_folder / "mask3.png")
        features_3_np = features_3.squeeze(0).cpu().numpy() if torch.is_tensor(features_3) else features_3.squeeze(0)
        feature_map_rgb, H, W = compute_pca_visualization(features_3_np, n_components)
        mask_3d = mask_3_np.reshape(H, W, 1).repeat(3, axis=2)
        masked_pca = (feature_map_rgb * mask_3d + 255 * (1 - mask_3d)).astype(np.uint8)
        masked_pca_pil = Image.fromarray(masked_pca)
        masked_pca_pil = masked_pca_pil.resize((256, 256), Image.NEAREST)
        masked_pca_pil.save(result_folder / "masked_pca3.png")
    # Save metadata
    metadata_path = result_folder / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    return result_folder, gen_path, img_1_path, img_2_path, img_3_path, metadata_path


if __name__ == "__main__":
    st.set_page_config(
        page_title="Custom Image Conditioning",
        page_icon="🖼️",
        layout="wide")
    
    st.title("Custom Image Conditioning with Dual Potentials")
    
    # Initialize session state
    if 'regen_counter' not in st.session_state:
        st.session_state.regen_counter = 0
    if 'features_computed' not in st.session_state:
        st.session_state.features_computed = False
    if 'image_files' not in st.session_state:
        st.session_state.image_files = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model selection
    available_models = {
        "SiT-LDM E2E-VAVAE (4M steps)": "pretrained/sit-ldm-e2e-vavae",
        "SiT-XL/2 256 REPA": "pretrained/SiT-XL-2-256-REPA.pt",
        "SiT-XL/2 256": "pretrained/SiT-XL-2-256.pt",
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(available_models.keys()),
        index=0
    )
    exp_path = available_models[selected_model_name]
    
    model, vae, latent_size, latents_scale, latents_bias = load_model(exp_path, device)
    
    with st.sidebar:
        st.header("Settings")
        
        # Folder selection
        st.subheader("Image Folder")
        folder_path = st.text_input("Image Folder Path", value="log/paper_vis/refs")
        
        if st.button("Load Images"):
            st.session_state.image_files = load_images_from_folder(folder_path)
            st.session_state.features_computed = False
            if len(st.session_state.image_files) > 0:
                st.success(f"Loaded {len(st.session_state.image_files)} images")
            else:
                st.error("No images found in folder")
        
        st.markdown("---")
        
        # Generation settings
        st.subheader("Generation Settings")
        sampling_mode = st.selectbox("Sampling Mode", ["ode", "sde"], index=0)
        num_steps = st.slider("Number of Steps", 1, 250, 50, step=1)
        heun = st.checkbox("Use Heun's Method", value=False)
        cfg_scale = st.slider("CFG Scale", 0.0, 10.0, 1.0, step=0.1)
        guidance_low = st.slider("Guidance Low", 0.0, 1.0, 0.0, step=0.1)
        guidance_high = st.slider("Guidance High", 0.0, 1.0, 1.0, step=0.1)
        path_type = st.selectbox("Path Type", ["linear", "cosine"], index=0)
        fix_noise = st.checkbox("Fix Noise for Generation", value=False)
        
        # Class conditioning
        y_options = {
            "Cat": 282,
            "Bus": 779,
            "Fireboat": 554,
            "angora rabbit":332,
            "Unconditional": 1000
        }
        selected_y_label = st.selectbox("Select ImageNet Class", list(y_options.keys()), index=4)
        y_class_id = y_options[selected_y_label]
        y = torch.tensor([y_class_id], device=device).to(torch.long)
        
        st.markdown("---")
        
        # Save settings
        st.subheader("Save Settings")
        output_folder = st.text_input("Output Folder", value="log/paper_vis/generated_results")
        
        sampling_args = dictdot(dict(
            num_steps=num_steps,
            heun=heun,
            cfg_scale=cfg_scale,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            path_type=path_type,
            sampling_mode=sampling_mode,
            gibbs=True
        ))
    
    # Load DINO model
    dino = load_dino(device)
    
    # Main content
    if len(st.session_state.image_files) == 0:
        st.info("Please load images from a folder using the sidebar settings")
    else:

        # Image selection sliders for three images
        col_slider1, col_slider2, col_slider3 = st.columns(3)
        with col_slider1:
            st.subheader("Image 1 Selection")
            idx_1 = st.slider("Select Image 1", 0, len(st.session_state.image_files)-1, 0, step=1, key="idx1")
        with col_slider2:
            st.subheader("Image 2 Selection")
            idx_2 = st.slider("Select Image 2", 0, len(st.session_state.image_files)-1, min(1, len(st.session_state.image_files)-1), step=1, key="idx2")
        with col_slider3:
            st.subheader("Image 3 Selection")
            idx_3 = st.slider("Select Image 3", 0, len(st.session_state.image_files)-1, min(2, len(st.session_state.image_files)-1), step=1, key="idx3")

        # Load and process selected images
        img_file_1 = st.session_state.image_files[idx_1]
        img_file_2 = st.session_state.image_files[idx_2]
        img_file_3 = st.session_state.image_files[idx_3]

        img_1_original = Image.open(img_file_1).convert('RGB')
        img_2_original = Image.open(img_file_2).convert('RGB')
        img_3_original = Image.open(img_file_3).convert('RGB')

        # Center crop and resize
        img_1 = center_crop_resize(img_1_original, 256)
        img_2 = center_crop_resize(img_2_original, 256)
        img_3 = center_crop_resize(img_3_original, 256)

        # Display images before computing features
        st.markdown("---")
        col_preview1, col_preview2, col_preview3 = st.columns(3)
        with col_preview1:
            st.subheader("Image 1 Preview")
            display_image(np.array(img_1))
            st.write(f"**File:** {img_file_1.name}")
        with col_preview2:
            st.subheader("Image 2 Preview")
            display_image(np.array(img_2))
            st.write(f"**File:** {img_file_2.name}")
        with col_preview3:
            st.subheader("Image 3 Preview")
            display_image(np.array(img_3))
            st.write(f"**File:** {img_file_3.name}")

        # Feature computation button
        st.markdown("---")
        compute_features_btn = st.button("🔍 Compute DINO Features", type="primary", use_container_width=True)
        if compute_features_btn:
            with st.spinner("Computing DINO features..."):
                # Preprocess for DINO
                img_1_dino = preprocess_image_for_dino(img_1, device)
                img_2_dino = preprocess_image_for_dino(img_2, device)
                img_3_dino = preprocess_image_for_dino(img_3, device)
                # Compute features
                st.session_state.features_1 = compute_features(dino, img_1_dino.squeeze(0), device, f"custom_{idx_1}")
                st.session_state.features_2 = compute_features(dino, img_2_dino.squeeze(0), device, f"custom_{idx_2}")
                st.session_state.features_3 = compute_features(dino, img_3_dino.squeeze(0), device, f"custom_{idx_3}")
                st.session_state.features_computed = True
                st.session_state.img_1_np = np.array(img_1)
                st.session_state.img_2_np = np.array(img_2)
                st.session_state.img_3_np = np.array(img_3)
            st.success("✅ Features computed successfully!")
        
        # Display images and potential settings
        if st.session_state.features_computed:
            st.markdown("---")
            
            # Add PCA components selector
            n_components = st.selectbox("Number of PCA Components", [1, 2, 3], index=2, 
                                       help="1: Grayscale, 2: RG channels, 3: Full RGB")
            

            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Image 1")
                display_image(st.session_state.img_1_np)
                st.write(f"**File:** {img_file_1.name}")
                st.markdown("### Potential Type 1")
                potential_type_1 = st.selectbox(
                    "Select Potential",
                    ["REPA (full)", "REPA (masked)", "Mean Feature Alignment", "Free Energy", "Transport", "Uncond"],
                    key="pot1"
                )
                if potential_type_1 == "REPA (full)":
                    lambda_1 = st.slider("Lambda (REPA 1)", 0.0, 100000.0, 50000.0, step=100.0, key="lambda1")
                elif potential_type_1 == "REPA (masked)":
                    lambda_1 = st.slider("Lambda (REPA 1)", 0.0, 100000.0, 50000.0, step=100.0, key="lambda1")
                    st.info("👆 Select mask region in the interactive feature map below")
                elif potential_type_1 == "Mean Feature Alignment":
                    lambda_1 = st.slider("Lambda (Mean 1)", 0.0, 100000.0, 50000.0, step=0.1, key="lambda1")
                elif potential_type_1 == "Free Energy":
                    lambda_1 = st.slider("Lambda (Free Energy 1)", 0.0, 20000.0, 3000.0, step=10.0, key="lambda1")
                    temperature_1 = st.slider("Temperature 1", 0.1, 10.0, 1.0, step=0.1, key="temp1")
                elif potential_type_1 == "Transport":
                    lambda_1 = st.slider("Lambda (Transport 1)", 0.0, 5000.0, 3000.0, step=10.0, key="lambda1")
                    eps_1 = st.slider("Epsilon 1", 0.01, 5.0, 1.0, step=0.01, key="eps1")
                    transport_rate_1 = st.slider("Transport Rate 1", 0.1, 5.0, 1.0, step=0.1, key="tr1")
                if potential_type_1 != "Uncond":
                    st.subheader("Interactive Feature Map 1")
                    display_interactive_feature_map(
                        st.session_state.features_1, 
                        original_image=st.session_state.img_1_np,
                        n_components=n_components,
                        key=0,
                        clear_button_key="clear_btn_1"
                    )
            with col2:
                st.subheader("Image 2")
                display_image(st.session_state.img_2_np)
                st.write(f"**File:** {img_file_2.name}")
                st.markdown("### Potential Type 2")
                potential_type_2 = st.selectbox(
                    "Select Potential",
                    ["REPA (full)", "REPA (masked)", "Mean Feature Alignment", "Free Energy", "Transport", "Uncond"],
                    key="pot2"
                )
                if potential_type_2 == "REPA (full)":
                    lambda_2 = st.slider("Lambda (REPA 2)", 0.0, 100000.0, 50000.0, step=100.0, key="lambda2")
                elif potential_type_2 == "REPA (masked)":
                    lambda_2 = st.slider("Lambda (REPA 2)", 0.0, 100000.0, 50000.0, step=100.0, key="lambda2")
                    st.info("👆 Select mask region in the interactive feature map below")
                elif potential_type_2 == "Mean Feature Alignment":
                    lambda_2 = st.slider("Lambda (Mean 2)", 0.0, 100000.0, 50000.0, step=0.1, key="lambda2")
                elif potential_type_2 == "Free Energy":
                    lambda_2 = st.slider("Lambda (Free Energy 2)", 0.0, 20000.0, 3000.0, step=10.0, key="lambda2")
                    temperature_2 = st.slider("Temperature 2", 0.1, 10.0, 1.0, step=0.1, key="temp2")
                elif potential_type_2 == "Transport":
                    lambda_2 = st.slider("Lambda (Transport 2)", 0.0, 5000.0, 3000.0, step=10.0, key="lambda2")
                    eps_2 = st.slider("Epsilon 2", 0.01, 5.0, 1.0, step=0.01, key="eps2")
                    transport_rate_2 = st.slider("Transport Rate 2", 0.1, 5.0, 1.0, step=0.1, key="tr2")
                if potential_type_2 != "Uncond":
                    st.subheader("Interactive Feature Map 2")
                    display_interactive_feature_map(
                        st.session_state.features_2, 
                        original_image=st.session_state.img_2_np,
                        n_components=n_components,
                        key=1,
                        clear_button_key="clear_btn_2"
                    )
            with col3:
                st.subheader("Image 3")
                display_image(st.session_state.img_3_np)
                st.write(f"**File:** {img_file_3.name}")
                st.markdown("### Potential Type 3")
                potential_type_3 = st.selectbox(
                    "Select Potential",
                    ["REPA (full)", "REPA (masked)", "Mean Feature Alignment", "Free Energy", "Transport", "Uncond"],
                    key="pot3"
                )
                if potential_type_3 == "REPA (full)":
                    lambda_3 = st.slider("Lambda (REPA 3)", 0.0, 100000.0, 50000.0, step=100.0, key="lambda3")
                elif potential_type_3 == "REPA (masked)":
                    lambda_3 = st.slider("Lambda (REPA 3)", 0.0, 100000.0, 50000.0, step=100.0, key="lambda3")
                    st.info("👆 Select mask region in the interactive feature map below")
                elif potential_type_3 == "Mean Feature Alignment":
                    lambda_3 = st.slider("Lambda (Mean 3)", 0.0, 100000.0, 50000.0, step=0.1, key="lambda3")
                elif potential_type_3 == "Free Energy":
                    lambda_3 = st.slider("Lambda (Free Energy 3)", 0.0, 20000.0, 3000.0, step=10.0, key="lambda3")
                    temperature_3 = st.slider("Temperature 3", 0.1, 10.0, 1.0, step=0.1, key="temp3")
                elif potential_type_3 == "Transport":
                    lambda_3 = st.slider("Lambda (Transport 3)", 0.0, 5000.0, 3000.0, step=10.0, key="lambda3")
                    eps_3 = st.slider("Epsilon 3", 0.01, 5.0, 1.0, step=0.01, key="eps3")
                    transport_rate_3 = st.slider("Transport Rate 3", 0.1, 5.0, 1.0, step=0.1, key="tr3")
                if potential_type_3 != "Uncond":
                    st.subheader("Interactive Feature Map 3")
                    display_interactive_feature_map(
                        st.session_state.features_3, 
                        original_image=st.session_state.img_3_np,
                        n_components=n_components,
                        key=2,
                        clear_button_key="clear_btn_3"
                    )
            
            # Generation section
            st.markdown("---")
            st.header("Generate Image")
            
            generate_btn = st.button("🎨 Generate with Multi-Potential", type="primary", use_container_width=True)
            if generate_btn:
                # Build potentials list
                potentials = []
                potential_names = []
                # Create potential 1
                if potential_type_1 != "Uncond":
                    if potential_type_1 == "REPA (full)":
                        pot1 = RepaPotential(
                            cond=st.session_state["feature_map_raw_0"].unsqueeze(0).to(device),
                            lamda=lambda_1
                        )
                        potential_names.append(f"REPA(full,λ={lambda_1})")
                    elif potential_type_1 == "REPA (masked)":
                        mask_1 = st.session_state.get("mask_0")
                        if mask_1 is not None:
                            pot1 = RepaPotential(
                                cond=st.session_state["feature_map_raw_0"].unsqueeze(0).to(device),
                                lamda=lambda_1,
                                mask=mask_1.to(device)
                            )
                            potential_names.append(f"REPA(masked,λ={lambda_1})")
                        else:
                            st.warning("No mask selected for Image 1. Using full REPA instead.")
                            pot1 = RepaPotential(
                                cond=st.session_state["feature_map_raw_0"].unsqueeze(0).to(device),
                                lamda=lambda_1
                            )
                            potential_names.append(f"REPA(full,λ={lambda_1})")
                    elif potential_type_1 == "Mean Feature Alignment":
                        pot1 = MeanFeatAlignment(
                            cond=st.session_state["feature_map_raw_0"].unsqueeze(0).mean(dim=-2, keepdim=True).to(device),
                            lamda=lambda_1
                        )
                        potential_names.append(f"MeanFeat(λ={lambda_1})")
                    elif potential_type_1 == "Free Energy":
                        pot1 = FreeEnergy(
                            cond=st.session_state["selected_features_tensor_0"].unsqueeze(-2).to(device),
                            lamda=lambda_1,
                            T=temperature_1
                        )
                        potential_names.append(f"FreeEnergy(λ={lambda_1},T={temperature_1})")
                    elif potential_type_1 == "Transport":
                        pot1 = TransportPotential(
                            cond=st.session_state["selected_features_tensor_0"].unsqueeze(-2).to(device),
                            lamda=lambda_1,
                            eps=eps_1,
                            transport_rate=transport_rate_1
                        )
                        potential_names.append(f"Transport(λ={lambda_1},ε={eps_1},r={transport_rate_1})")
                    potentials.append(pot1)
                # Create potential 2
                if potential_type_2 != "Uncond":
                    if potential_type_2 == "REPA (full)":
                        pot2 = RepaPotential(
                            cond=st.session_state["feature_map_raw_1"].unsqueeze(0).to(device),
                            lamda=lambda_2
                        )
                        potential_names.append(f"REPA(full,λ={lambda_2})")
                    elif potential_type_2 == "REPA (masked)":
                        mask_2 = st.session_state.get("mask_1")
                        if mask_2 is not None:
                            pot2 = RepaPotential(
                                cond=st.session_state["feature_map_raw_1"].unsqueeze(0).to(device),
                                lamda=lambda_2,
                                mask=mask_2.to(device)
                            )
                            potential_names.append(f"REPA(masked,λ={lambda_2})")
                        else:
                            st.warning("No mask selected for Image 2. Using full REPA instead.")
                            pot2 = RepaPotential(
                                cond=st.session_state["feature_map_raw_1"].unsqueeze(0).to(device),
                                lamda=lambda_2
                            )
                            potential_names.append(f"REPA(full,λ={lambda_2})")
                    elif potential_type_2 == "Mean Feature Alignment":
                        pot2 = MeanFeatAlignment(
                            cond=st.session_state["feature_map_raw_1"].unsqueeze(0).mean(dim=-2, keepdim=True).to(device),
                            lamda=lambda_2
                        )
                        potential_names.append(f"MeanFeat(λ={lambda_2})")
                    elif potential_type_2 == "Free Energy":
                        pot2 = FreeEnergy(
                            cond=st.session_state["selected_features_tensor_1"].unsqueeze(-2).to(device),
                            lamda=lambda_2,
                            T=temperature_2
                        )
                        potential_names.append(f"FreeEnergy(λ={lambda_2},T={temperature_2})")
                    elif potential_type_2 == "Transport":
                        pot2 = TransportPotential(
                            cond=st.session_state["selected_features_tensor_1"].unsqueeze(-2).to(device),
                            lamda=lambda_2,
                            eps=eps_2,
                            transport_rate=transport_rate_2
                        )
                        potential_names.append(f"Transport(λ={lambda_2},ε={eps_2},r={transport_rate_2})")
                    potentials.append(pot2)
                # Create potential 3
                if potential_type_3 != "Uncond":
                    if potential_type_3 == "REPA (full)":
                        pot3 = RepaPotential(
                            cond=st.session_state["feature_map_raw_2"].unsqueeze(0).to(device),
                            lamda=lambda_3
                        )
                        potential_names.append(f"REPA(full,λ={lambda_3})")
                    elif potential_type_3 == "REPA (masked)":
                        mask_3 = st.session_state.get("mask_2")
                        if mask_3 is not None:
                            pot3 = RepaPotential(
                                cond=st.session_state["feature_map_raw_2"].unsqueeze(0).to(device),
                                lamda=lambda_3,
                                mask=mask_3.to(device)
                            )
                            potential_names.append(f"REPA(masked,λ={lambda_3})")
                        else:
                            st.warning("No mask selected for Image 3. Using full REPA instead.")
                            pot3 = RepaPotential(
                                cond=st.session_state["feature_map_raw_2"].unsqueeze(0).to(device),
                                lamda=lambda_3
                            )
                            potential_names.append(f"REPA(full,λ={lambda_3})")
                    elif potential_type_3 == "Mean Feature Alignment":
                        pot3 = MeanFeatAlignment(
                            cond=st.session_state["feature_map_raw_2"].unsqueeze(0).mean(dim=-2, keepdim=True).to(device),
                            lamda=lambda_3
                        )
                        potential_names.append(f"MeanFeat(λ={lambda_3})")
                    elif potential_type_3 == "Free Energy":
                        pot3 = FreeEnergy(
                            cond=st.session_state["selected_features_tensor_2"].unsqueeze(-2).to(device),
                            lamda=lambda_3,
                            T=temperature_3
                        )
                        potential_names.append(f"FreeEnergy(λ={lambda_3},T={temperature_3})")
                    elif potential_type_3 == "Transport":
                        pot3 = TransportPotential(
                            cond=st.session_state["selected_features_tensor_2"].unsqueeze(-2).to(device),
                            lamda=lambda_3,
                            eps=eps_3,
                            transport_rate=transport_rate_3
                        )
                        potential_names.append(f"Transport(λ={lambda_3},ε={eps_3},r={transport_rate_3})")
                    potentials.append(pot3)
                # Create multi-potential or single potential
                if len(potentials) == 0:
                    st.warning("All potentials are set to 'Uncond'. Generating unconditional image.")
                    final_potential = None
                elif len(potentials) == 1:
                    final_potential = potentials[0]
                else:
                    final_potential = MultiPotential(potentials, secondary_potential_guidance_threshold=0.0)
                # Generate
                with st.spinner("Generating image with multi-potential conditioning..."):
                    if fix_noise:
                        seed = 0
                        z = get_constant_noise(model.in_channels, latent_size, device)
                    else:
                        seed = hash((idx_1, idx_2, idx_3, st.session_state.regen_counter)) % (2**31)
                        z = None
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    img_gen, _ = generate_image(
                        final_potential, y, model, vae, latent_size, 
                        latents_scale, latents_bias, sampling_args, device, 
                        fixed_noise=z
                    )
                    # Convert to uint8
                    img_gen = img_gen.transpose(1, 2, 0)
                    img_gen = np.clip(img_gen, -1, 1)
                    img_gen = (img_gen + 1) / 2
                    img_gen = (img_gen * 255).astype(np.uint8)
                    st.session_state.img_gen = img_gen
                    st.session_state.generated = True
                    st.session_state.last_seed = seed
                    st.session_state.regen_counter += 1
                    # Prepare metadata
                    metadata = {
                        "timestamp": datetime.now().isoformat(),
                        "model": selected_model_name,
                        "image_1": str(img_file_1),
                        "image_2": str(img_file_2),
                        "image_3": str(img_file_3),
                        "potential_1": potential_type_1,
                        "potential_2": potential_type_2,
                        "potential_3": potential_type_3,
                        "potential_params_1": {},
                        "potential_params_2": {},
                        "potential_params_3": {},
                        "class_conditioning": selected_y_label,
                        "class_id": int(y_class_id),
                        "seed": seed,
                        "sampling_mode": sampling_mode,
                        "num_steps": num_steps,
                        "cfg_scale": float(cfg_scale),
                        "guidance_low": float(guidance_low),
                        "guidance_high": float(guidance_high),
                        "path_type": path_type,
                        "heun": heun,
                        "fix_noise": fix_noise,
                    }
                    # Add potential-specific params to metadata
                    if potential_type_1 == "REPA (full)" or potential_type_1 == "REPA (masked)":
                        metadata["potential_params_1"]["lambda"] = float(lambda_1)
                        metadata["potential_params_1"]["masked"] = (potential_type_1 == "REPA (masked)")
                    elif potential_type_1 == "Mean Feature Alignment":
                        metadata["potential_params_1"]["lambda"] = float(lambda_1)
                    elif potential_type_1 == "Free Energy":
                        metadata["potential_params_1"]["lambda"] = float(lambda_1)
                        metadata["potential_params_1"]["temperature"] = float(temperature_1)
                    elif potential_type_1 == "Transport":
                        metadata["potential_params_1"]["lambda"] = float(lambda_1)
                        metadata["potential_params_1"]["epsilon"] = float(eps_1)
                        metadata["potential_params_1"]["transport_rate"] = float(transport_rate_1)
                    if potential_type_2 == "REPA (full)" or potential_type_2 == "REPA (masked)":
                        metadata["potential_params_2"]["lambda"] = float(lambda_2)
                        metadata["potential_params_2"]["masked"] = (potential_type_2 == "REPA (masked)")
                    elif potential_type_2 == "Mean Feature Alignment":
                        metadata["potential_params_2"]["lambda"] = float(lambda_2)
                    elif potential_type_2 == "Free Energy":
                        metadata["potential_params_2"]["lambda"] = float(lambda_2)
                        metadata["potential_params_2"]["temperature"] = float(temperature_2)
                    elif potential_type_2 == "Transport":
                        metadata["potential_params_2"]["lambda"] = float(lambda_2)
                        metadata["potential_params_2"]["epsilon"] = float(eps_2)
                        metadata["potential_params_2"]["transport_rate"] = float(transport_rate_2)
                    if potential_type_3 == "REPA (full)" or potential_type_3 == "REPA (masked)":
                        metadata["potential_params_3"]["lambda"] = float(lambda_3)
                        metadata["potential_params_3"]["masked"] = (potential_type_3 == "REPA (masked)")
                    elif potential_type_3 == "Mean Feature Alignment":
                        metadata["potential_params_3"]["lambda"] = float(lambda_3)
                    elif potential_type_3 == "Free Energy":
                        metadata["potential_params_3"]["lambda"] = float(lambda_3)
                        metadata["potential_params_3"]["temperature"] = float(temperature_3)
                    elif potential_type_3 == "Transport":
                        metadata["potential_params_3"]["lambda"] = float(lambda_3)
                        metadata["potential_params_3"]["epsilon"] = float(eps_3)
                        metadata["potential_params_3"]["transport_rate"] = float(transport_rate_3)
                    st.session_state.last_metadata = metadata
                st.success("✅ Generation completed!")
            
            # Display and save results
            if 'img_gen' in st.session_state and st.session_state.generated:
                st.markdown("---")
                st.header("Generated Result")
                col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                with col_res1:
                    st.subheader("Input 1")
                    display_image(st.session_state.img_1_np)
                with col_res2:
                    st.subheader("Input 2")
                    display_image(st.session_state.img_2_np)
                with col_res3:
                    st.subheader("Input 3")
                    display_image(st.session_state.img_3_np)
                with col_res4:
                    st.subheader("Generated")
                    display_image(st.session_state.img_gen)
                    st.write(f"**Seed:** {st.session_state.last_seed}")
                # Save button
                st.markdown("---")
                if st.button("💾 Save Results", type="secondary", use_container_width=True):
                    try:
                        mask_1 = st.session_state.get("mask_0")
                        mask_2 = st.session_state.get("mask_1")
                        mask_3 = st.session_state.get("mask_2")
                        result_folder, gen_path, img1_path, img2_path, img3_path, meta_path = save_generation_result(
                            st.session_state.img_gen,
                            st.session_state.img_1_np,
                            st.session_state.img_2_np,
                            st.session_state.last_metadata,
                            output_folder,
                            mask_1=mask_1,
                            mask_2=mask_2,
                            features_1=st.session_state.features_1,
                            features_2=st.session_state.features_2,
                            n_components=n_components,
                            img_3=st.session_state.img_3_np,
                            mask_3=mask_3,
                            features_3=st.session_state.features_3
                        )
                        files_list = ["generated.png", "input1.png", "input2.png", "input3.png", "metadata.json"]
                        if mask_1 is not None:
                            files_list.extend(["mask1.png", "masked_pca1.png"])
                        if mask_2 is not None:
                            files_list.extend(["mask2.png", "masked_pca2.png"])
                        if mask_3 is not None:
                            files_list.extend(["mask3.png", "masked_pca3.png"])
                        st.success(f"✅ Results saved to folder: `{result_folder}`")
                        st.info(f"📁 Folder: `{result_folder.name}`\n\n📷 Files: {', '.join(f'`{f}`' for f in files_list)}")
                    except Exception as e:
                        st.error(f"Error saving results: {str(e)}")
        else:
            st.info("👆 Please click 'Compute DINO Features' to proceed with generation")
