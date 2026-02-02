import streamlit as st
from streamlit_drawable_canvas import st_canvas
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np
import torch
import json
import gc
from dictdot import dictdot
from einops import rearrange, repeat
import time
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from sklearn.decomposition import PCA # Added for analysis
import scipy.stats as stats # Added for analysis
import io

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Ensure your sampler imports are correct here
from samplers import euler_sampler, euler_maruyama_sampler, compute_single_step_drift_euler, compute_single_step_drift_euler_maruyama
from models.sit import SiT_models
from models.autoencoder import vae_models
from utils import load_encoders, load_sit_and_vae
from dataset_not_h5 import ImageNetDataset
from dataset_coco import SimpleCOCODataset
# from batched_quality_evaluation import Aesthetics, ImageReward, PickScore


preprocess_raw_img = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

@st.cache_resource
def load_imagenet(root):
    dataset_dino = ImageNetDataset(
        root=root,
        split='train',
        transform=preprocess_raw_img,
    )
    dataset_vis = ImageNetDataset(
        root=root,
        split='train',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            ]))
    return dataset_dino, dataset_vis

@st.cache_resource
def load_coco(root, split='train', year='2017'):
    dataset_dino = SimpleCOCODataset(
        root=root,
        split=split,
        year=year,
        image_size=256,
        transform=preprocess_raw_img,
    )
    dataset_vis = SimpleCOCODataset(
        root=root,
        split=split,
        year=year,
        image_size=256,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
        ]),
    )
    return dataset_dino, dataset_vis

def display_image(img):
    st.image(img, width=256)

@st.cache_data
def compute_pca_visualization(feature_map_raw, n_components=3):
    """Compute PCA for visualization with caching
    Args:
        feature_map_raw: numpy array of shape (L, D)
        n_components: number of principal components to use (1, 2, or 3)
    """
    L, D = feature_map_raw.shape
    H = W = int(np.sqrt(L))
    
    # PCA for visualization
    feature_map_centered = feature_map_raw - feature_map_raw.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(feature_map_centered, full_matrices=False)
    
    # Handle different numbers of principal components
    n_components = min(n_components, Vt.shape[0])
    feature_map_pca = np.dot(feature_map_centered, Vt.T[:, :n_components])
    
    # Normalize
    feature_map_pca = feature_map_pca - feature_map_pca.min()
    feature_map_pca = feature_map_pca / (feature_map_pca.max() + 1e-8)
    
    # Convert to RGB based on number of components
    if n_components == 1:
        # Grayscale: replicate single component to all 3 channels
        feature_map_rgb = np.repeat(feature_map_pca, 3, axis=-1)
    elif n_components == 2:
        # Use first 2 components for R and G, set B to 0
        feature_map_rgb = np.zeros((feature_map_pca.shape[0], 3))
        feature_map_rgb[:, :2] = feature_map_pca
    else:
        # Use all 3 components for RGB
        feature_map_rgb = feature_map_pca
    
    feature_map_rgb = feature_map_rgb.reshape(H, W, 3)
    
    # Convert to uint8 for display
    feature_map_rgb = (feature_map_rgb * 255).astype(np.uint8)
    
    return feature_map_rgb, H, W

def compute_flow_feature_heatmap(features, sink_point, eps=1.0, step_size=1.0):
    """
    Computes a single step of feature flow towards a sink point using numpy.

    Args:
        features (np.ndarray): The feature map, shape (L, D).
        sink_point (np.ndarray): The target feature (sink), shape (D,).
        eps (float): A small value for numerical stability and controlling gradient magnitude.
        step_size (float): The step size for the feature update.

    Returns:
        np.ndarray: The updated features, shape (L, D).
    """
    features_normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    sink_point = sink_point.flatten()
    sink_point_normalized = sink_point / np.linalg.norm(sink_point)
    # Ensure sink_point is a 1D array

    # d is the cosine similarity between each feature vector and the sink point
    L, D = features_normalized.shape
    H = W = int(np.sqrt(L))
    d = np.clip(np.dot(features_normalized, sink_point_normalized), -1.0, 1.0)  # Shape (L,)
    # d = np.dot(features_normalized, sink_point_normalized) # Shape (L,)
    # print(d)
    # v is the component of sink_point orthogonal to features
    v = sink_point - d[:, np.newaxis] * features
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v = v / (v_norm + 1e-8)  # Normalize v
    
    # r is the angle between features_normalized and sink_point_normalized
    r = np.arccos(d)  # Shape (L,)
    r2 = r**2
    grad_mag = 2.0 * r / eps * np.exp(-r2 / eps)
    angle = step_size * grad_mag
    ca = np.cos(angle)[:, np.newaxis]
    ca = np.reshape(ca, (H, W))
    
    # Normalize similarities to [0, 1] for color mapping
    min_sim = ca.min()
    max_sim = ca.max()
    if max_sim == min_sim: # Avoid division by zero if all similarities are the same
        normalized_feature_flow = np.zeros_like(ca)
    else:
        normalized_feature_flow = (ca - min_sim) / (max_sim - min_sim)
    
    # Convert to RGB using a colormap (e.g., viridis)
    cmap = plt.get_cmap('viridis')
    # feature_map_rgb = cmap(normalized_feature_flow)[:, :, :3] # Take only RGB channels
    feature_map_rgb = cmap((1+ca)/2.0)[:, :, :3] # Take only RGB channels
    
    # Convert to uint8 for display
    feature_map_rgb = (feature_map_rgb * 255).astype(np.uint8)
    
    return feature_map_rgb

def display_feature_flow_heatmap(feature_map, target_feature):
    feature_map_raw = feature_map.squeeze(0).cpu().numpy()  # (L, D)
    target_feature = target_feature.squeeze(0).cpu().numpy()  # (D,)
    L, D = feature_map_raw.shape
    
    feature_map_rgb = compute_flow_feature_heatmap(feature_map_raw, target_feature)
    
    st.image(feature_map_rgb, width=256)

def compute_feature_similarity_heatmap(feature_map_raw, target_feature):
    """Compute feature similarity heatmap with caching
    Args:
        feature_map_raw: numpy array of shape (L, D)
        target_feature: numpy array of shape (D,)
    """
    L, D = feature_map_raw.shape
    H = W = int(np.sqrt(L))
    # PCA for visualization
    # normalize the featuremap 
    target_feature_normalized = target_feature / np.linalg.norm(target_feature)
    
    # Compute cosine similarity
    # Ensure feature_map_raw is normalized patch-wise
    feature_map_normalized = feature_map_raw / np.linalg.norm(feature_map_raw, axis=1, keepdims=True)
    
    # Compute dot product (cosine similarity)
    patch_similarities = np.dot(feature_map_normalized, target_feature_normalized) # Shape: (L,)
    
    # Reshape to 2D for heatmap
    patch_similarities_2d = patch_similarities.reshape(H, W)
    
    # Normalize similarities to [0, 1] for color mapping
    min_sim = patch_similarities_2d.min()
    max_sim = patch_similarities_2d.max()
    if max_sim == min_sim: # Avoid division by zero if all similarities are the same
        normalized_similarities = np.zeros_like(patch_similarities_2d)
    else:
        normalized_similarities = (patch_similarities_2d - min_sim) / (max_sim - min_sim)
    
    # Convert to RGB using a colormap (e.g., viridis)
    cmap = plt.get_cmap('viridis')
    feature_map_rgb = cmap(normalized_similarities)[:, :, :3] # Take only RGB channels
    
    # Convert to uint8 for display
    feature_map_rgb = (feature_map_rgb * 255).astype(np.uint8)
    
    return feature_map_rgb, H, W

def display_feature_similarity_heatmap(feature_map, target_feature):
    feature_map_raw = feature_map.squeeze(0).cpu().numpy()  # (L, D)
    target_feature = target_feature.squeeze(0).cpu().numpy()  # (D,)
    L, D = feature_map_raw.shape
    
    # Compute PCA (cached)
    feature_map_rgb, H, W = compute_feature_similarity_heatmap(feature_map_raw, target_feature)
    
    st.image(feature_map_rgb, width=256)

def display_interactive_feature_selector(feature_map, n_components=3):
    feature_map_raw = feature_map.squeeze(0).cpu().numpy()  # (L, D)
    L, D = feature_map_raw.shape
    
    # Store raw features in session state for downstream tasks
    st.session_state.feature_map_raw = feature_map_raw
    st.session_state.selected_features_tensor = None
    st.session_state.selected_pixel = (0, 0)
    # Compute PCA (cached)
    feature_map_rgb, H, W = compute_pca_visualization(feature_map_raw, n_components)
    
    # Create interactive plot
    fig = go.Figure(data=go.Image(z=feature_map_rgb))
    fig.update_layout(
        width=600,
        height=600,
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(constrain="domain"),
    )
    
    # Display the plot and capture clicks
    selected_point = st.plotly_chart(fig, use_container_width=False, on_select="rerun", key=f"feature_map")
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None

    # Manual pixel selection with number inputs
    col1, col2 = st.columns(2)
    with col1:
        x_coord = st.number_input(f"X coordinate (0-{W-1})", min_value=0, max_value=W-1, value=0, key="x_coord")
    with col2:
        y_coord = st.number_input(f"Y coordinate (0-{H-1})", min_value=0, max_value=H-1, value=0, key="y_coord")
    
    if st.button("Select Pixel"):
        st.session_state.selected_pixel = (int(y_coord), int(x_coord))
        # Calculate linear index from 2D coordinates
        pixel_idx = int(y_coord) * W + int(x_coord)
        st.session_state.selected_features = feature_map_raw[pixel_idx]
        st.session_state.mask = None  # Clear mask for pixel selection
    
    # Display selected pixel features
    if st.session_state.selected_pixel is not None and st.session_state.selected_features is not None:
        y, x = st.session_state.selected_pixel
        selected_features = st.session_state.selected_features
        
        st.success(f"Selected Pixel: ({x}, {y})")
        
        # Convert to torch tensor for downstream tasks
        st.session_state.selected_features_tensor = torch.from_numpy(selected_features).float()

def display_clip_score(gen_img):
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-large-patch14")

    def calculate_clip_score(images, prompts):
        images_int = images.astype("uint8")
        clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)
    
    # get the current conditioning class and the current representative image type and compute the clip score of the current image for a prompt "class on representative image type"
    if st.session_state.get('conditioning_class') is not None and st.session_state.get('representative_image_type') is not None:
        conditioning_class = st.session_state.conditioning_class
        representative_image_type = st.session_state.representative_image_type
        
        prompt = f"An image of a {conditioning_class} on {representative_image_type.lower()}"
        
        # Convert generated image to PIL Image and then to numpy array
        score = calculate_clip_score(gen_img[None, ...], [prompt])
        st.metric(label=f"CLIP Score for '{prompt}'", value=score)

def display_aesthetics_score(gen_img, device):
    if st.session_state.get('conditioning_class') is not None and st.session_state.get('representative_image_type') is not None:
        aesthetics = Aesthetics(device)
        images_int = gen_img.astype("uint8")
        aesthetics_score = aesthetics.compute_score(Image.fromarray(images_int))
        st.metric(label="Aesthetics Score", value=round(float(aesthetics_score), 4))
    
def display_image_reward_score(gen_img, device):
    if st.session_state.get('conditioning_class') is not None and st.session_state.get('representative_image_type') is not None:
        conditioning_class = st.session_state.conditioning_class
        representative_image_type = st.session_state.representative_image_type
        image_reward = ImageReward(device)
        images_int = gen_img.astype("uint8")
        prompt = f"An image of a {conditioning_class} on {representative_image_type.lower()}"
        image_reward_score = image_reward.compute_score(Image.fromarray(images_int), txt=prompt)
        st.metric(label="Image Reward Score", value=round(float(image_reward_score), 4))

def display_pick_score(gen_img, device):
    if st.session_state.get('conditioning_class') is not None and st.session_state.get('representative_image_type') is not None:
        conditioning_class = st.session_state.conditioning_class
        representative_image_type = st.session_state.representative_image_type
        pick = PickScore(device)
        images_int = gen_img.astype("uint8")
        prompt = f"An image of a {conditioning_class} on {representative_image_type.lower()}"
        pick_score = pick.compute_score(Image.fromarray(images_int), txt=prompt)
        st.metric(label="Pick Score", value=round(float(pick_score), 4))

def display_representative_feature_map(dino=None, device=None, n_components=3):
    #Make a selector titled "Background" that chooses between "Grass" and "Water" and then selects the corresponding directory in app/assets/representative_images
    representative_image_type = st.selectbox("Background", ["Grass", "Water"], index=0)
    st.session_state.representative_image_type = representative_image_type
    representative_image_dir = os.path.join("app", "assets", "representative_images", representative_image_type.lower())
    
    # List all .png files in the directory
    image_files = [f for f in os.listdir(representative_image_dir) if f.endswith('.png')]
    
    if not image_files:
        st.warning(f"No .png images found in {representative_image_dir}")
    else:
        # Select an image from the list
        selected_rep_image_file = st.selectbox("Select Representative Image", image_files)
        # Load the selected image
        rep_image_path = os.path.join(representative_image_dir, selected_rep_image_file)
        rep_image_pil = Image.open(rep_image_path).convert("RGB")
        
        st.image(rep_image_pil, caption=f"Selected Representative Image: {selected_rep_image_file}", width=256)
        
        # Preprocess the representative image for DINOv2
        rep_image_tensor = preprocess_raw_img(rep_image_pil)
        
        # Compute features for the representative image
        with torch.no_grad():
            rep_features = dino.forward_features(rep_image_tensor.unsqueeze(0).to(device))['x_norm_patchtokens']
    
    feature_map_rgb, H, W = compute_pca_visualization(rep_features.squeeze(0).cpu().numpy(), n_components)
    
    # Create interactive plot
    fig = go.Figure(data=go.Image(z=feature_map_rgb))
    fig.update_layout(
        width=600,
        height=600,
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(constrain="domain"),
    )
    st.plotly_chart(fig, use_container_width=False, on_select="rerun", key="feature_map")
    
    # Add selector to choose between Average Feature and Specific Feature
    representative_feature_selection = st.radio(
        "Select conditioning feature from representative image:",
        ["Average Feature", "Specific Feature", "Whole Feature"],
        index=0,
        key="rep_feature_selection"
    )

    if representative_feature_selection == "Specific Feature":
        # Manual pixel selection with number inputs for representative image
        col1_rep, col2_rep = st.columns(2)
        with col1_rep:
            x_coord_rep = st.number_input(f"X coordinate (0-{W-1})", min_value=0, max_value=W-1, value=0, key="x_coord_rep")
        with col2_rep:
            y_coord_rep = st.number_input(f"Y coordinate (0-{H-1})", min_value=0, max_value=H-1, value=0, key="y_coord_rep")
        
        if st.button("Select Specific Pixel from Representative Image"):
            st.session_state.selected_pixel = (int(y_coord_rep), int(x_coord_rep))
            # Calculate linear index from 2D coordinates
            pixel_idx_rep = int(y_coord_rep) * W + int(x_coord_rep)
            st.session_state.selected_features = rep_features.squeeze(0).cpu().numpy()[pixel_idx_rep]
            st.session_state.selected_features_tensor = torch.from_numpy(st.session_state.selected_features).float()
            st.session_state.mask = None # Clear mask for pixel selection
            st.session_state.box_coords = None
            st.success(f"Selected Specific Pixel ({x_coord_rep}, {y_coord_rep}) from '{selected_rep_image_file}' as conditioning.")
    elif representative_feature_selection == "Average Feature": # Average Feature selected
        if st.button("Use Average Feature from Representative Image"):
            st.session_state.selected_pixel = (0, 0)
        # Average the features to get a single representative feature vector
        avg_rep_features = torch.linalg.matrix_norm(rep_features)
        avg_rep_features = rep_features.mean(dim=1).squeeze(0) # Shape: (D,)
        st.session_state.selected_features_tensor = avg_rep_features
        st.session_state.mask = None # No mask for representative image conditioning
        st.session_state.box_coords = None
        st.session_state.selected_pixel = None
        
        st.success(f"Using features from '{selected_rep_image_file}' as conditioning.")
    elif representative_feature_selection == "Whole Feature":
        if st.button("Use Whole Feature"):
            st.session_state.selected_features_tensor = rep_features
            st.session_state.mask = None # No mask for representative image conditioning
            st.session_state.box_coords = None
            st.session_state.selected_pixel = None
            st.success(f"Using all features from '{selected_rep_image_file}' as conditioning.")
        

def display_interactive_feature_map(feature_map, original_image=None, n_components=3, key=None, clear_button_key=None):
    """Interactive feature map with PCA visualization and pixel selection
    Args:
        feature_map: tensor of shape (1, L, D) where L is number of patches
        original_image: PIL Image or numpy array of the original image for canvas drawing
        n_components: number of principal components to use for visualization (1, 2, or 3)
        key: optional key for multiple feature maps, None for backward compatibility
    """
    feature_map_raw = feature_map.squeeze(0).cpu().numpy()  # (L, D)
    L, D = feature_map_raw.shape
    
    # Helper function to get session state key names
    def get_key(name):
        return f"{name}_{key}" if key is not None else name
    
    # Store raw features in session state for downstream tasks
    st.session_state[get_key("feature_map_raw")] = torch.from_numpy(feature_map_raw)
    # st.session_state.selected_features_tensor = None
    # Compute PCA (cached)
    feature_map_rgb, H, W = compute_pca_visualization(feature_map_raw, n_components)
    
    # Create interactive plot
    fig = go.Figure(data=go.Image(z=feature_map_rgb))
    fig.update_layout(
        width=600,
        height=600,
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(constrain="domain"),
    )
    
    # Display the plot and capture clicks
    plot_key = f"feature_map_{key}" if key is not None else "feature_map"
    st.plotly_chart(fig, use_container_width=False, on_select="rerun", key=plot_key)
    
    # Initialize session state for selected pixel
    if get_key("selected_pixel") not in st.session_state:
        st.session_state[get_key("selected_pixel")] = None
    if get_key("selected_features") not in st.session_state:
        st.session_state[get_key("selected_features")] = None
    if get_key("mask") not in st.session_state:
        st.session_state[get_key("mask")] = None
    if get_key("box_coords") not in st.session_state:
        st.session_state[get_key("box_coords")] = None
    
    # Selection mode
    st.subheader("Selection Mode")
    mode_key = f"selection_mode_{key}" if key is not None else "selection_mode"
    selection_mode = st.radio("Select conditioning type:", ["Single Pixel", "Box Region", "PCA Threshold"], horizontal=True, key=mode_key)

    if selection_mode == "Single Pixel":
        # Manual pixel selection with number inputs
        col1, col2 = st.columns(2)
        with col1:
            x_key = f"x_coord_{key}" if key is not None else "x_coord"
            x_coord = st.number_input(f"X coordinate (0-{W-1})", min_value=0, max_value=W-1, value=0, key=x_key)
        with col2:
            y_key = f"y_coord_{key}" if key is not None else "y_coord"
            y_coord = st.number_input(f"Y coordinate (0-{H-1})", min_value=0, max_value=H-1, value=0, key=y_key)
        
        btn_key = f"select_pixel_{key}" if key is not None else "select_pixel"
        if st.button("Select Pixel", key=btn_key):
            st.session_state[get_key("selected_pixel")] = (int(y_coord), int(x_coord))
            # Calculate linear index from 2D coordinates
            pixel_idx = int(y_coord) * W + int(x_coord)
            st.session_state[get_key("selected_features")] = feature_map_raw[pixel_idx]
            st.session_state[get_key("mask")] = None  # Clear mask for pixel selection
        # Display selected pixel features
        if st.session_state[get_key("selected_pixel")] is not None and st.session_state[get_key("selected_features")] is not None:
            y, x = st.session_state[get_key("selected_pixel")]
            selected_features = st.session_state[get_key("selected_features")]
            st.success(f"Selected Pixel: ({x}, {y})")
            # Convert to torch tensor for downstream tasks
            st.session_state[get_key("selected_features_tensor")] = torch.from_numpy(selected_features).float()
        # Display selected pixels
        if get_key("selected_pixels") in st.session_state and st.session_state[get_key("selected_features")] is not None and isinstance(st.session_state[get_key("selected_features")], list):
            (y1, x1), (y2, x2) = st.session_state[get_key("selected_pixels")]
            st.success(f"Selected Pixels: (x1={x1}, y1={y1}), (x2={x2}, y2={y2})")
    elif selection_mode == "Box Region":  # Box Region
        st.write("Draw a box on the image to select a region:")

        # Convert numpy array to PIL Image if needed
        if isinstance(original_image, np.ndarray):
            canvas_image = Image.fromarray(original_image)
        else:
            canvas_image = original_image
        
        # Create drawable canvas with original image as background
        canvas_key = f"canvas_rect_{key}" if key is not None else "canvas_rect"
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",  # Semi-transparent red fill
            stroke_width=3,
            stroke_color="#FF0000",  # Red stroke
            background_image=canvas_image,
            height=canvas_image.size[1],
            width=canvas_image.size[0],
            drawing_mode="rect",
            key=canvas_key,
        )
        
        # Process canvas drawing
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                # Get the last drawn rectangle
                rect = objects[-1]
                if rect["type"] == "rect":
                    # Extract coordinates (in canvas space)
                    left = rect["left"]
                    top = rect["top"]
                    width = rect["width"]
                    height = rect["height"]
                    
                    # Convert from canvas coordinates (512x512) to feature map coordinates (HxW)
                    # Canvas shows the full image, feature map is HxW patches
                    x_min = int((left / canvas_image.size[0]) * W)
                    y_min = int((top / canvas_image.size[1]) * H)
                    x_max = int(((left + width) / canvas_image.size[0]) * W)
                    y_max = int(((top + height) / canvas_image.size[1]) * H)
                    
                    # Clamp to valid range
                    x_min = max(0, min(W-1, x_min))
                    y_min = max(0, min(H-1, y_min))
                    x_max = max(0, min(W-1, x_max))
                    y_max = max(0, min(H-1, y_max))
                    
                    # Ensure valid box
                    if x_max > x_min and y_max > y_min:
                        box_coords = (x_min, y_min, x_max, y_max)
                        mask = compute_box_mask(box_coords, H, W)
                        st.session_state[get_key("mask")] = mask
                        st.session_state[get_key("box_coords")] = box_coords
                        
                        # Compute average features in the box region
                        mask_np = mask.squeeze(0).numpy()
                        masked_features = feature_map_raw * mask_np[:, None]
                        avg_features = masked_features.sum(axis=0) / (mask_np.sum() + 1e-8)
                        st.session_state[get_key("selected_features_tensor")] = torch.from_numpy(avg_features).float()
                        
                        st.success(f"Box selected: ({x_min}, {y_min}) to ({x_max}, {y_max}) - {int(mask_np.sum())} pixels")
        
        # Show current selection info 
        if get_key("mask") in st.session_state and st.session_state[get_key("mask")] is not None and get_key("box_coords") in st.session_state:
            x_min, y_min, x_max, y_max = st.session_state[get_key("box_coords")]
            mask_np = st.session_state[get_key("mask")].squeeze(0).numpy()
            st.info(f"Current selection: ({x_min}, {y_min}) to ({x_max}, {y_max}) - {int(mask_np.sum())} pixels")
            
            # Option to clear selection
            if st.button("Clear Selection", key=clear_button_key):
                st.session_state[get_key("mask")] = None
                st.session_state[get_key("box_coords")] = None
                st.rerun()
    elif selection_mode == "PCA Threshold":
        st.write("Create a binary mask based on PCA component threshold:")
        # Threshold slider
        slider_key = f"pca_threshold_slider_{key}" if key is not None else None
        threshold = st.slider("Threshold value", 0.0, 1.0, 0.5, step=0.01,
                    help="Pixels with PCA values above this threshold will be selected",
                    key=slider_key)
        
        # Create binary mask
        binary_mask = compute_mask_from_pca(feature_map_raw, threshold)
        
        # Visualize the mask (flip vertically to match image orientation)
        mask_2d = binary_mask.reshape(H, W)
        fig_mask = go.Figure(data=go.Heatmap(z=mask_2d[::-1], colorscale='Greys', showscale=False))
        fig_mask.update_layout(
            width=300,
            height=300,
            title="Binary Mask",
            xaxis=dict(scaleanchor="y", constrain="domain"),
            yaxis=dict(constrain="domain"),
        )
        pca_mask_key = f"pca_mask_{key}" if key is not None else "pca_mask"
        st.plotly_chart(fig_mask, use_container_width=False, key=pca_mask_key)
        
        # Apply mask button
        apply_mask_key = f"apply_threshold_mask_{key}" if key is not None else "apply_threshold_mask"
        if st.button("Apply Threshold Mask", key=apply_mask_key):
            mask_tensor = torch.from_numpy(binary_mask).float().unsqueeze(0)
            st.session_state[get_key("mask")] = mask_tensor
            st.session_state[get_key("box_coords")] = None  # Clear box coords for threshold mask
            
            # Compute average features in the masked region
            masked_features = feature_map_raw * binary_mask[:, None]
            avg_features = masked_features.sum(axis=0) / (binary_mask.sum() + 1e-8)
            st.session_state[get_key("selected_features_tensor")] = torch.from_numpy(avg_features).float()
            
            st.success(f"Threshold mask applied: {int(binary_mask.sum())} pixels selected")
        
        # Show current threshold mask info
        if get_key("mask") in st.session_state and st.session_state[get_key("mask")] is not None and (get_key("box_coords") not in st.session_state or st.session_state[get_key("box_coords")] is None):
            mask_np = st.session_state[get_key("mask")].squeeze(0).numpy()
            if mask_np.sum() > 0:  # Only show if it's a threshold mask (non-box mask)
                st.info(f"Current threshold mask: {int(mask_np.sum())} pixels selected")
                
                # Option to clear selection
                if st.button("Clear Threshold Mask", key=clear_button_key):
                    st.session_state[get_key("mask")] = None
                    st.session_state[get_key("box_coords")] = None
                    st.rerun()

def compute_mask_from_pca(features, threshold):
    """
    Computes a binary foreground mask from feature maps using PCA.
    Handles a batch of feature maps.
    """
    # Ensure input is a numpy array on CPU
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    L, D = features.shape
    H = W = int(np.sqrt(L))
    n_components = 1  # Always use the first principal component for the mask

    feature_map = features  # Shape: (L, D)

    # Compute PCA
    feature_map_centered = feature_map - feature_map.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(feature_map_centered, full_matrices=False)
    
    # Project onto the first principal component
    pca_component = np.dot(feature_map_centered, Vt.T[:, :n_components]).flatten()

    # Normalize to [0, 1]
    pca_normalized = (pca_component - pca_component.min()) / (pca_component.max() - pca_component.min() + 1e-8)

    # Threshold at the median to create a binary mask
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

    # Stack masks into a single numpy array of shape (B, L)
    return binary_mask

def compute_box_mask(box_coords, H, W):
    """Create a binary mask from box coordinates
    Args:
        box_coords: tuple of (x_min, y_min, x_max, y_max)
        H, W: height and width of the feature map
    Returns:
        mask: torch tensor of shape (1, H*W) with 1s inside box, 0s outside
    """
    x_min, y_min, x_max, y_max = box_coords
    mask = np.zeros((H, W))
    mask[y_min:y_max+1, x_min:x_max+1] = 1
    mask = mask.reshape(-1)  # Flatten to (H*W,)
    return torch.from_numpy(mask).float().unsqueeze(0)  # Shape: (1, H*W)

def get_selected_features():
    """Helper function to retrieve selected features for downstream tasks"""
    if 'selected_features' in st.session_state and st.session_state.selected_features is not None:
        return {
            'features': st.session_state.selected_features,
            'features_tensor': st.session_state.selected_features_tensor,
            'pixel_coords': st.session_state.selected_pixel,
            'full_feature_map': st.session_state.feature_map_raw
        }
    return None

@st.cache_resource
def load_dino(device):
    return load_encoders("dinov2-vit-b",device,256)[0][0]

@st.cache_data
def compute_features(_dino, _img_tensor, _device, idx):
    """Compute DINO features with caching per image index"""
    with torch.no_grad():
        feature = _dino.forward_features(_img_tensor.unsqueeze(0).to(_device))['x_norm_patchtokens']
    return feature

@st.cache_resource
def load_model(exp_path, device, checkpoint_name=None):
    """
    Load a SiT model and VAE with Streamlit caching.
    
    This is a wrapper around load_sit_and_vae that provides caching
    and maintains backward compatibility with the app interface.
    
    Args:
        exp_path: Path to experiment directory (e.g., 'pretrained/sit-ldm-e2e-vavae')
        device: Device to load models on
        checkpoint_name: Optional checkpoint filename (e.g., '4000000.pt'). If None, uses latest.
    """
    print(f"Loading model from {exp_path}...")
    
    # Load config if it exists
    config = None
    config_path = os.path.join(exp_path, "args.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = dictdot(json.load(f))
    
    # Determine checkpoint path
    if checkpoint_name:
        checkpoint_path = os.path.join(exp_path, "checkpoints", checkpoint_name)
    else:
        # Find latest checkpoint
        checkpoints_dir = os.path.join(exp_path, "checkpoints")
        if os.path.exists(checkpoints_dir):
            checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
            if checkpoints:
                # Sort by number in filename
                checkpoints.sort(key=lambda x: int(x.replace('.pt', '')))
                checkpoint_path = os.path.join(checkpoints_dir, checkpoints[-1])
            else:
                raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
        else:
            # Assume exp_path is the checkpoint file itself
            checkpoint_path = exp_path
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    model, vae, latents_scale, latents_bias, checkpoint, latent_size, loaded_config = load_sit_and_vae(
        checkpoint_path, device, config=config
    )
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return model, vae, latent_size, latents_scale, latents_bias



def generate_image(potential, y, model, vae, latent_size, latents_scale, latents_bias, args, device, fixed_noise=None, batch_size=1):
    # Sample inputs:
    latent_size = latent_size if isinstance(latent_size, tuple) else (latent_size, latent_size)
    if fixed_noise is not None:
        z = fixed_noise.to(device)
    else:
        z = torch.randn(batch_size, model.in_channels, latent_size[0], latent_size[1], device=device, requires_grad=True)
    if y is None:
        y = torch.randint(0, 1, (batch_size,), device=device)

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
    )
    # if args.sampling_mode == "sde":
    #     x, intermediates = euler_maruyama_sampler(**sampling_kwargs)
    # else:
    #     x, intermediates = euler_sampler(**sampling_kwargs)
    
    if args.sampling_mode == "sde":
        x = euler_maruyama_sampler(**sampling_kwargs)
        intermediates=None
    else:
        x = euler_sampler(**sampling_kwargs)
        intermediates=None
    x = x.to(torch.float32)
    
    with torch.no_grad():
        x_decoded = vae.decode(x / latents_scale + latents_bias).sample
    
    # Return both the decoded image and the latent intermediates
    return x_decoded.squeeze().cpu().numpy(), intermediates

def get_single_step_drift(potential, x_cur, y, model, latent_size, args, device):
    # Sample inputs:
    latent_size = latent_size if isinstance(latent_size, tuple) else (latent_size, latent_size)
    if y is None:
        y = torch.randint(0, 1, (1,), device=device)

    # Sample images:
    sampling_kwargs = dict(
        model=model, 
        x_cur=x_cur,
        y=y,
        potential=potential,
        t_cur=0, 
        cfg_scale=args.cfg_scale,
        guidance_low=args.guidance_low,
        guidance_high=args.guidance_high,
    )
    if args.sampling_mode == "sde":
        x = compute_single_step_drift_euler_maruyama(**sampling_kwargs).to(torch.float32)
    else:
        x = compute_single_step_drift_euler(**sampling_kwargs).to(torch.float32)

    return x.squeeze().cpu().numpy()

def display_single_step_drift(potential, x_cur, y, model, latent_size, args, device):
    visual = get_single_step_drift(potential, x_cur, y, model, latent_size, args, device)
    return visual

@st.cache_data
def get_constant_noise(in_channels, latent_size, device):
    return torch.randn(1, in_channels, latent_size, latent_size, device=device, requires_grad=True)

def plot_latent_trajectory(intermediates):
    """
    1. Latent Trajectory (PCA)
    Projects high-dim latents to 2D PCA space.
    intermediates: Tensor [Steps, C, H, W]
    """
    # Flatten: [Steps, Features]
    steps = intermediates.shape[0]
    flat_states = intermediates.reshape(steps, -1).numpy()
    
    # Fit PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(flat_states)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Line
    fig.add_trace(go.Scatter(
        x=pca_result[:, 0], y=pca_result[:, 1],
        mode='lines+markers',
        name='Trajectory',
        marker=dict(size=6, color=np.arange(steps), colorscale='Viridis', showscale=True, colorbar=dict(title="Step")),
        line=dict(color='gray', width=1, dash='dot')
    ))
    
    # Start/End annotations
    fig.add_trace(go.Scatter(x=[pca_result[0,0]], y=[pca_result[0,1]], mode='markers', marker=dict(color='red', symbol='x', size=12), name='Start'))
    fig.add_trace(go.Scatter(x=[pca_result[-1,0]], y=[pca_result[-1,1]], mode='markers', marker=dict(color='green', symbol='star', size=12), name='End'))
    
    fig.update_layout(title="Latent Space Trajectory (PCA)", xaxis_title="PC1", yaxis_title="PC2", width=600, height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_energy_drift(intermediates):
    """
    2. Drift Magnitude (Energy)
    Computes L2 norm of difference between steps.
    """
    diffs = torch.diff(intermediates, dim=0) # [Steps-1, C, H, W]
    # L2 norm per step
    norms = torch.norm(diffs.reshape(diffs.shape[0], -1), dim=1).numpy()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=norms, mode='lines', fill='tozeroy', name='Drift Magnitude'))
    fig.update_layout(
        title="Drift Energy (Step-wise Rate of Change)",
        xaxis_title="Step",
        yaxis_title="L2 Norm of Update (|x_t+1 - x_t|)",
        width=600, height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_spatial_force(intermediates):
    """
    3. Spatial Force Map
    Heatmap of total updates per pixel.
    """
    # Calculate difference between steps to get the magnitude of updates
    diffs = torch.diff(intermediates, dim=0).abs() 
    
    # Robustly average across all dimensions except the last two (Height, Width)
    # This ensures we get a 2D [H, W] map regardless if input is [Steps, C, H, W] or [Steps, H, W]
    dims_to_reduce = tuple(range(diffs.dim() - 2))
    heatmap = diffs.mean(dim=dims_to_reduce)
    
    # Ensure it's on CPU and numpy
    heatmap_np = heatmap.detach().cpu().numpy()
    
    fig = px.imshow(
        heatmap_np, 
        color_continuous_scale='Magma', 
        title="Spatial Update Intensity (Heatmap)",
        labels=dict(x="Width", y="Height", color="Intensity")
    )
    fig.update_layout(width=500, height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_spectral_evolution(intermediates):
    """
    4. Spectral Evolution (FFT)
    """
    def get_radial_profile(img):
        # img must be 2D [H, W]
        y, x = np.indices((img.shape))
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
        
        tbin = np.bincount(r.ravel(), img.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / (nr + 1e-8)
        return radialprofile

    # Robust dimension reduction
    # intermediates shape is likely [Steps, Batch, Channels, H, W] or [Steps, Channels, H, W]
    # We want to preserve Dim 0 (Steps) and the last 2 Dims (H, W)
    # We take the mean of everything in between
    
    dims_to_reduce = tuple(range(1, intermediates.ndim - 2))
    if dims_to_reduce:
        features = intermediates.mean(dim=dims_to_reduce).numpy() # Shape: [Steps, H, W]
    else:
        features = intermediates.numpy()

    # Compute FFT per step
    spectra = []
    
    for i in range(features.shape[0]):
        # spatial_slice is now guaranteed to be [H, W]
        spatial_slice = features[i] 
        
        f = np.fft.fft2(spatial_slice)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        
        profile = get_radial_profile(magnitude_spectrum)
        spectra.append(profile[:len(profile)//2]) # Keep low to high freq
        
    spectra_map = np.array(spectra).T # [Freq, Steps]
    
    fig = px.imshow(
        spectra_map, 
        labels=dict(x="Time Step", y="Frequency (Low -> High)", color="Log Magnitude"),
        title="Spectral Evolution (FFT)",
        color_continuous_scale='Jet',
        origin='lower'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_vector_field(intermediates):
    """
    5. Vector Field Slice
    Visualizes flow of a specific crop.
    """
    # Ensure intermediates is on CPU
    intermediates = intermediates.cpu()
    
    # Handle dimensions: we want [Steps, Channels, H, W]
    # If [Steps, 1, C, H, W], squeeze the batch dim
    if intermediates.ndim == 5 and intermediates.shape[1] == 1:
        intermediates = intermediates.squeeze(1)
        
    steps = intermediates.shape[0]
    target_step = min(steps - 2, int(steps * 0.2)) 
    
    # Get current and next state for the target step
    # Shape is now expected to be [Channels, H, W]
    state_curr = intermediates[target_step]
    state_next = intermediates[target_step + 1]
    
    # Extract the first channel for the background image [H, W]
    # We use index 0 for the channel
    current_img = state_curr[0].numpy()
    
    # Calculate flow (difference)
    diff = (state_next - state_curr).numpy()
    
    # Setup U (x-flow) and V (y-flow)
    # Use Channel 0 for U. If we have >1 channel, use Channel 1 for V, else reuse Ch 0
    u_full = diff[0]
    if diff.shape[0] > 1:
        v_full = diff[1]
    else:
        v_full = diff[0] # Fallback for 1-channel latents
    
    # Downsample for visualization (quiver plots get messy at full res)
    from skimage.transform import resize
    scale = 32 # Resolution of the vector grid
    
    # Resize background, U, and V to the grid size
    bg_resized = resize(current_img, (scale, scale), anti_aliasing=True)
    u = resize(u_full, (scale, scale), anti_aliasing=True)
    v = resize(v_full, (scale, scale), anti_aliasing=True)
    
    # Create grid
    x, y = np.meshgrid(np.arange(scale), np.arange(scale))
    
    fig = plt.figure(figsize=(8, 8))
    plt.title(f"Latent Flow Field (Step {target_step}, Channels 0 & 1)")
    
    # Display background (inverted colormap often looks better for latents)
    plt.imshow(bg_resized, cmap='gray', alpha=0.5, origin='upper')
    
    # Plot vectors
    # pivot='mid' centers the arrow on the grid point
    plt.quiver(x, y, u, v, color='red', scale=None, pivot='mid')
    
    # Clean up axis
    plt.axis('off')
    
    st.pyplot(fig)

def create_diffusion_gif(intermediates, vae, latents_scale, latents_bias, fps=20):
    """
    Decodes each step of the diffusion process into an RGB image and creates a GIF.
    """
    frames = []
    device = next(vae.parameters()).device
    
    # Ensure intermediates are on CPU initially to save GPU VRAM
    intermediates = intermediates.cpu()
    
    # Remove batch dim if present: [Steps, 1, C, H, W] -> [Steps, C, H, W]
    if intermediates.ndim == 5:
        intermediates = intermediates.squeeze(1)
        
    steps = intermediates.shape[0]
    
    for i in range(steps):
        with torch.no_grad():
            # 1. Get single latent step, move to GPU, and CAST TO FLOAT32
            # The .float() here fixes the "Input type (double) and bias type (float)" error
            z = intermediates[i].to(device).unsqueeze(0).float() 
            
            # 2. Denormalize based on SiT/DiT standard training logic
            z_unscaled = z / latents_scale + latents_bias
            
            # 3. Decode with VAE
            x_decoded = vae.decode(z_unscaled).sample
            
            # 4. Process for display
            img = x_decoded.squeeze(0).cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, -1, 1)
            img = (img + 1) / 2
            img_uint8 = (img * 255).astype(np.uint8)
            
            frames.append(Image.fromarray(img_uint8))

    # Save to buffer
    buffer = io.BytesIO()
    frames[0].save(
        buffer, 
        format="GIF", 
        save_all=True, 
        append_images=frames[1:], 
        duration=1000//fps, 
        loop=0
    )
    return buffer.getvalue()