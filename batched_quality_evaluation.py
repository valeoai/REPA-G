import os
import re
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
import json
import inflect
# --- Distributed Processing Imports ---
from accelerate import Accelerator

# --- Library Imports ---
from transformers import AutoProcessor, AutoModel
from aesthetic_predictor import predict_aesthetic 
from torchvision.transforms import Normalize
import torch.nn.functional as F
from dataset import CustomINH5ClassDataset
from torchvision.transforms.functional import to_pil_image
# ==================================================================================
# --- METRICS ---
# ==================================================================================

class BatchClipScore:
    def __init__(self, device):
        self.device = device
        self.model = AutoModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    def compute_batch(self, images, texts):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            img_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            return (img_embeds * txt_embeds).sum(dim=-1).cpu().numpy().tolist()

class BatchPickScore:
    def __init__(self, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("yuvalkirstain/PickScore_v1")
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
    def compute_batch(self, images, texts):
        i_inputs = self.processor(images=images, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        t_inputs = self.processor(text=texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        with torch.no_grad():
            i_emb = self.model.get_image_features(**i_inputs); i_emb /= i_emb.norm(dim=-1, keepdim=True)
            t_emb = self.model.get_text_features(**t_inputs); t_emb /= t_emb.norm(dim=-1, keepdim=True)
            return (i_emb * t_emb).sum(dim=-1).cpu().numpy().tolist()

class BatchDinoDiscriminability:
    def __init__(self, device):
        self.device = device
        print(f"Loading DINOv2 (ViT-B/14) via Torch Hub on {device}...")
        
        # Load Model
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        # Clean up head as per your generation script
        del self.model.head
        self.model.head = torch.nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Constants for Preprocessing
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

    def preprocess_raw_image(self, x):
        """
        Mirroring the generation script's preprocessing for 'dinov2'.
        x: Tensor [B, C, H, W] with values 0-255
        """
        resolution = x.shape[-1]
        
        # 1. Scale to [0, 1]
        x = x / 255.0
        
        # 2. Normalize
        # Normalize expects [C, H, W], but works on [B, C, H, W] nicely in recent torchvision
        x = Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)(x)
        
        # 3. Interpolate
        # target_size logic: 224 * (resolution // 256)
        target_size = 224 * (resolution // 256)
        x = F.interpolate(x, size=target_size, mode='bicubic', align_corners=False)
        
        return x

    def compute_batch(self, gen_images, anchor_images, target_features, image_ids, experiment_paths):
        # Helper to convert PIL lists to Batch Tensor (0-255)
        def to_batch_tensor(pil_list):
            # Convert PIL -> Tensor (C, H, W) [0-255, uint8] -> Float -> Stack
            # We convert to Float here to allow division in preprocess
            tensors = []
            for img in pil_list:
                # Convert to numpy/tensor manually to ensure 0-255 range preservation
                # (ToTensor() often scales to 0-1 automatically, which we don't want yet)
                arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
                tensors.append(arr)
            return torch.stack(tensors).to(self.device)

        # 1. Prepare Batch Tensors
        gen_raw = to_batch_tensor(gen_images)
        anc_raw = to_batch_tensor(anchor_images)

        # 2. Apply "preprocess_raw_image" logic
        gen_input = self.preprocess_raw_image(gen_raw)
        anc_input = self.preprocess_raw_image(anc_raw)

        # 3. Get Spatial Features
        def get_spatial_feats(inputs):
            with torch.no_grad():
                return self.model.forward_features(inputs)['x_norm_patchtokens']

        gen_feats = get_spatial_feats(gen_input)
        anc_feats = get_spatial_feats(anc_input)
        tgt_feats = target_features.to(self.device).unsqueeze(1)
        # 2. Normalize
        gen_feats = gen_feats / gen_feats.norm(dim=-1, keepdim=True)
        anc_feats = anc_feats / anc_feats.norm(dim=-1, keepdim=True)
        tgt_feats = tgt_feats / tgt_feats.norm(dim=-1, keepdim=True)

        # 3. Compute Similarity Maps
        sim_gen_anc = (gen_feats * anc_feats).sum(dim=-1) 
        sim_gen_tgt = (gen_feats * tgt_feats).sum(dim=-1) 

        # 4. Difference & Vote
        diff_map = (sim_gen_tgt - sim_gen_anc)
        
        # --- METRIC 1: Raw Vote (Original) ---
        # Fraction of patches closer to Target
        is_target_closer = (diff_map > 0).float()
        raw_votes = is_target_closer.mean(dim=-1).cpu().numpy().tolist()
        
        # --- METRIC 2: Group-Balanced Vote (New) ---
        # Mean(Sim_Target_Group) - Mean(Sim_Anchor_Group)
        # This handles cases where the object is small (few target patches) 
        # vs background (many anchor patches).
        
        # Create masks
        mask_target = (diff_map > 0)
        mask_anchor = ~mask_target
        
        # Average similarity within the "Target Group"
        # We clamp denominator to 1e-6 to avoid division by zero if no patches match
        mean_sim_target_group = (sim_gen_tgt * mask_target).sum(dim=-1) / (mask_target.sum(dim=-1) + 1e-6)
        mean_sim_target_group = mean_sim_target_group.cpu().numpy().tolist()
        
        # Average similarity within the "Anchor Group"
        mean_sim_anchor_group = (sim_gen_anc * mask_anchor).sum(dim=-1) / (mask_anchor.sum(dim=-1) + 1e-6)
        mean_sim_anchor_group = mean_sim_anchor_group.cpu().numpy().tolist()

        # --- METRIC 3: Max Similarity ---
        max_similarity = torch.maximum(sim_gen_tgt, sim_gen_anc).mean(dim=-1).cpu().numpy().tolist()

        # 6. Save Heatmaps (.npz) - OVERWRITES EXISTING
        heatmap_paths = []
        n_patches = diff_map.shape[1]
        grid_size = int(np.sqrt(n_patches))
        diff_map_np = diff_map.cpu().numpy()
        
        for i, d_map_flat in enumerate(diff_map_np):
            img_id = image_ids[i]
            exp_path = Path(experiment_paths[i])
            
            save_dir = exp_path / "dino_comparison_heatmaps"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            heatmap_2d = d_map_flat.reshape(grid_size, grid_size)
            fname = save_dir / f"{img_id}_heatmap.npz"
            
            np.savez_compressed(fname, heatmap=heatmap_2d)
            heatmap_paths.append(str(fname))

        return raw_votes, heatmap_paths, max_similarity, mean_sim_target_group, mean_sim_anchor_group

class BatchAesthetics:
    def __init__(self, device):
        self.device = device
    def compute_batch(self, images):
        scores = []
        try:
            s = predict_aesthetic(images)
            for score in s: scores.append(float(score))
        except:
            for img in images: scores.append(float(predict_aesthetic(img)))
        return scores

def get_balanced_imagenet_class_names():
    with open("assets/imagenet_class_to_names.json", "r") as f:
        class_mapping = json.load(f)
    return class_mapping
# ==================================================================================
# --- PARSING STRATEGIES ---
# ==================================================================================

class ParsingStrategy(ABC):
    def __init__(self):
        self.class_map = get_balanced_imagenet_class_names()
        self.p = inflect.engine()
    @abstractmethod
    def parse(self, experiment_path: Path, image_path: Path):
        pass

    def _get_prompt(self, class_id, suffix=None):
        class_name = self.class_map[str(class_id)]
        class_name_with_article = self.p.a(class_name)
        if suffix:
            # suffix example: "on_yellow_grass"
            # 1. Replace underscores -> "on yellow grass"
            # 2. Split -> ["on", "yellow", "grass"]
            # 3. Slice [1:] to remove preposition -> ["yellow", "grass"]
            words = suffix.replace("_", " ").strip().split(" ")[1:]
            
            # 4. JOIN them back into a string -> "yellow grass"
            background_text = " ".join(words)
            
            # 5. Apply article -> "a yellow grass"
            background_with_article = self.p.a(background_text)
            
            return f"A photo of {class_name_with_article} with {background_with_article} background.", class_name
        return f"A photo of a {class_name}.", class_name

class SinglePotentialGridSearchStrategy(ParsingStrategy):
    def __init__(self):
        super().__init__()
        self.dir_regex = re.compile(r"low(\d+\.?\d*)_high[\d\.]+_potential-([a-zA-Z0-9-]+).*?class_(\d+)_source_([a-zA-Z0-9-]+)_sample_(\d+)")
        self.file_regex = re.compile(r"(\d+\.?\d*)_other_(\d+\.?\d*)\.png")

    def parse(self, experiment_path, image_path):
        dir_match = self.dir_regex.search(experiment_path.name)
        file_match = self.file_regex.search(image_path.name)
        if not dir_match or not file_match: return None

        class_id = int(dir_match.group(3))
        prompt, class_name = self._get_prompt(class_id)

        return {
            "guidance_low": dir_match.group(1),
            "potential_type": dir_match.group(2),
            "class_id": class_id,
            "class_name": class_name,
            "source_type": dir_match.group(4),
            "sample_id": int(dir_match.group(5)),
            "lambda": float(file_match.group(1)),
            "other_param": float(file_match.group(2))
        }

class MultiPotentialGridSearchStrategy(ParsingStrategy):
    def __init__(self):
        super().__init__()
        self.dir_regex = re.compile(r"MultiPot_.*?_([a-zA-Z0-9-]+)_ancLam")
        self.file_regex = re.compile(r"anchor_class_(\d+)_anchor_sample_(\d+)_lam(\d+\.?\d*)_(?:eps(\d+\.?\d*)_|temp(\d+\.?\d*)_)?guide_low_(\d+\.?\d*)_(.*?)_rep_class_(\d+)_rep_sample_(\d+)\.png")

    def parse(self, experiment_path, image_path):
        dir_match = self.dir_regex.search(experiment_path.name)
        file_match = self.file_regex.search(image_path.name)
        if not dir_match or not file_match: return None

        eps_val = file_match.group(4)
        temp_val = file_match.group(5)
        other_val = 0.0
        if eps_val: other_val = float(eps_val)
        elif temp_val: other_val = float(temp_val)

        class_id = int(file_match.group(1))
        rep_class_id = int(file_match.group(8))
        rep_sample_id = int(file_match.group(9))
        source_suffix = file_match.group(7)
        prompt, class_name = self._get_prompt(class_id, suffix=source_suffix)

        return {
            "potential_type": dir_match.group(1),
            "class_id": class_id,
            "class_name": class_name,
            "sample_id": int(file_match.group(2)),
            "rep_class_id": rep_class_id,
            "rep_sample_id": rep_sample_id,
            "lambda": float(file_match.group(3)),
            "other_param": other_val,
            "guidance_low": file_match.group(6),
            "source_type": source_suffix,
            "prompt": prompt,
        }

class MultiPotentialGenerateStrategy(ParsingStrategy):
    """
    Mode: multi_potential_gen
    Parses 'generate' output structure.
    File example: method_multipot-free-energy_ac_670_as_999_tc_98_ts_127688_prompt_in_water.png
    """
    def __init__(self):
        super().__init__()
        self.dir_regex = re.compile(r"([a-zA-Z0-9-]+-[a-zA-Z0-9-]+)_.*?_steps_(\d+).*?_pl_(\d+\.?\d*)_po_(\d+\.?\d*)(?:_([a-zA-Z0-9]+))?")
        # Group 4 is target_class (tc), Group 5 is target_sample (ts)
        self.file_regex = re.compile(r"method_([a-zA-Z0-9-]+-[a-zA-Z0-9-]+)_ac_(\d+)_as_(\d+)_tc_(\d+)_ts_(\d+)_prompt_(.*?)\.png")

    def parse(self, experiment_path, image_path):
        dir_match = self.dir_regex.search(experiment_path.name)
        file_match = self.file_regex.search(image_path.name)
        if not dir_match or not file_match: return None

        method_name = dir_match.group(1)
        steps = int(dir_match.group(2))
        pot_lambda = float(dir_match.group(3))
        pot_other = float(dir_match.group(4))
        mode = dir_match.group(5) if dir_match.group(5) else "default"

        class_id = int(file_match.group(2))
        sample_id = int(file_match.group(3))
        rep_class_id = int(file_match.group(4))
        rep_sample_id = int(file_match.group(5))
        source_suffix = file_match.group(6)

        prompt, class_name = self._get_prompt(class_id, suffix=source_suffix)

        return {
            "potential_type": method_name, 
            "class_id": class_id,
            "class_name": class_name,
            "sample_id": sample_id,
            "rep_class_id": rep_class_id,
            "rep_sample_id": rep_sample_id,
            "lambda": pot_lambda,
            "other_param": pot_other,
            "steps": steps,            
            "sampling_mode": mode,     
            "guidance_low": "0.0", 
            "source_type": source_suffix,
            "prompt": prompt,
        }

def get_parsing_strategy(mode):
    if mode == 'single_potential_gs':
        return SinglePotentialGridSearchStrategy()
    elif mode == 'multi_potential_gs':
        return MultiPotentialGridSearchStrategy()
    elif mode == 'multi_potential_gen':
        return MultiPotentialGenerateStrategy()
    else:
        raise ValueError(f"Unknown mode: {mode}")

# ==================================================================================
# --- DATASET & MAIN ---
# ==================================================================================

class EvaluationDataset(Dataset):
    def __init__(self, items, dataset_root=None, target_features_path=None, imagenet_data_root=None):
        self.items = items
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.target_features = {} 
        if target_features_path:
            p = Path(target_features_path)
            if p.exists() and p.is_dir():
                print(f"Loading target features from directory: {p}")
                pattern = re.compile(r"^class_(?P<class_id>\d+)_sample_(?P<sample_id>\d+)_(?P<prompt>.+)\.pt$")
                for feat_file in p.glob("*.pt"):
                    match = pattern.search(feat_file.name)
                    if match:
                        cid = int(match.group('class_id'))
                        sid = int(match.group('sample_id'))
                        try:
                            self.target_features[(cid, sid)] = torch.load(feat_file, map_location="cpu")
                        except Exception as e:
                            print(f"Error loading {feat_file}: {e}")
        self.imagenet_dataset = CustomINH5ClassDataset(imagenet_data_root)
    def __len__(self):
        return len(self.items)

    def _get_image_path(self, class_id, sample_id):
        if self.dataset_root is None: return None
        path = self.dataset_root / str(class_id) / f"{sample_id}.png"
        if not path.exists():
             path = self.dataset_root / f"class_{class_id}_sample_{sample_id}.png"
        return path

    def __getitem__(self, idx):
        item = self.items[idx]
        gen_image = Image.open(item['file_path']).convert("RGB")

        anchor_image_index = self.imagenet_dataset.get_global_index(item['class_id'], item['sample_id'])
        anchor_image, _ = self.imagenet_dataset[anchor_image_index]
        anchor_image = to_pil_image(anchor_image)
        
        target_key = (item.get('rep_class_id'), item.get('rep_sample_id'))
        if target_key in self.target_features:
            target_feat = self.target_features[target_key]
        else:
            raise FileNotFoundError(f"Target file {target_key} not found")
        
        # Unique ID for saving heatmaps
        unique_id = Path(item['file_path']).stem

        return {
            "gen_image": gen_image, "anchor_image": anchor_image, "target_feature": target_feat,
            "prompt": item['prompt'], "id": unique_id, "metadata": item
        }

def custom_collate(batch):
    gen_imgs = [x['gen_image'] for x in batch]
    anc_imgs = [x['anchor_image'] for x in batch]
    tgt_feats = torch.stack([x['target_feature'] for x in batch]) 
    prompts = [x['prompt'] for x in batch]
    ids = [x['id'] for x in batch]
    metas = [x['metadata'] for x in batch]
    return gen_imgs, anc_imgs, tgt_feats, prompts, metas, ids

def evaluate(args):
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"Distributed evaluation on {accelerator.num_processes} GPUs.")
        print(f"Mode: {args.mode}")

    parser_strategy = get_parsing_strategy(args.mode)
    experiments = list(Path(args.save_dir).glob("*"))
    data_items = []
    
    if accelerator.is_main_process: print("Indexing files...")
    
    specific_experiments = []
    if args.specify_experiments:
        specific_experiments = [c.strip() for c in args.specify_experiments.split(',') if c.strip()]

    for experiment_path in experiments:
        if not experiment_path.is_dir(): continue
        if specific_experiments and experiment_path.name not in specific_experiments: continue
        
        image_paths = list(Path(experiment_path).glob("images/*/*.png"))
        if not image_paths: image_paths = list(Path(experiment_path).glob("images/*.png"))

        for image_path in image_paths:
            meta = parser_strategy.parse(experiment_path, image_path)
            if meta:
                meta['experiment_path'] = str(experiment_path) # Store path for DINO saving
                meta['experiment_name'] = experiment_path.name
                meta['file_path'] = str(image_path)
                meta['prompt'] = meta.get('prompt', "") 
                if not meta['prompt']: pass 
                data_items.append(meta)

    if accelerator.is_main_process: print(f"Found {len(data_items)} images.")

    dataset = EvaluationDataset(
        data_items, 
        dataset_root=args.dataset_root, 
        target_features_path=args.target_features_path,
        imagenet_data_root=args.dataset_root,
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=custom_collate
    )
    dataloader = accelerator.prepare(dataloader)

    if accelerator.is_main_process: print("Loading models...")
    m_clip = BatchClipScore(device)
    m_pick = BatchPickScore(device)

    if accelerator.is_main_process:
        print("Rank 0: Checking/Downloading DINOv2 model...")
        # This call ensures the files are present in the cache
        torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    
    # 2. Block all other processes until Rank 0 is done
    accelerator.wait_for_everyone()

    m_dino = BatchDinoDiscriminability(device) 

    local_rows = []
    progress_bar = tqdm(dataloader, desc="Eval", disable=not accelerator.is_local_main_process)
    
    for gen_imgs, anc_imgs, tgt_feats, prompts, batch_metadata, batch_ids in progress_bar:
        try:
            clip_scores = m_clip.compute_batch(gen_imgs, prompts)
            pick_scores = m_pick.compute_batch(gen_imgs, prompts)
            # Extract Experiment Paths for the current batch
            batch_exp_paths = [m['experiment_path'] for m in batch_metadata]
            
            # Compute Dino Discriminability & Save Heatmaps
            votes, heatmap_paths, max_sims, mean_sim_tgt, mean_sim_ancr = m_dino.compute_batch(gen_imgs, anc_imgs, tgt_feats, batch_ids, batch_exp_paths)
            for i in range(len(gen_imgs)):
                meta = batch_metadata[i]
                meta.update({
                    "clip_score": clip_scores[i],
                    "pick_score": pick_scores[i],
                    "dino_discriminability_vote": votes[i],
                    "max_similarity_measure": max_sims[i],
                    "similarity_heatmap_path": heatmap_paths[i],
                    "mean_target_sim": mean_sim_tgt[i],
                    "mean_anchor_sim": mean_sim_ancr[i],
                    "aesthetic_score": 0.0 # Disabled
                })
                local_rows.append(meta)
        except Exception as e:
            print(f"Error on rank {accelerator.process_index}: {e}")
            continue

    partial_filename = f"partial_results_rank_{accelerator.process_index}.csv"
    if local_rows: pd.DataFrame(local_rows).to_csv(partial_filename, index=False)
    
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("Merging results...")
        all_dfs = []
        for i in range(accelerator.num_processes):
            fname = f"partial_results_rank_{i}.csv"
            if os.path.exists(fname):
                all_dfs.append(pd.read_csv(fname))
                os.remove(fname)
        
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            output_csv_path = Path(args.save_dir) / "grid_search_scores_distributed_with_dino_2.csv"
            final_df.to_csv(output_csv_path, index=False)
            print(f"Saved: {output_csv_path}")
        else:
            print("No results found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="Path to generated samples")
    parser.add_argument("--dataset_root", type=str, default=None, help="Path to original Anchor images")
    parser.add_argument("--target_features_path", type=str, default=None, help="Path to folder containing .pt target features")
    parser.add_argument("--mode", type=str, required=True, choices=["single_potential_gs", "multi_potential_gs", "multi_potential_gen"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--specify_experiments", type=str, default=None)
    args = parser.parse_args()

    evaluate(args)