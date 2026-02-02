"""
COCO Dataset with ImageNet-style preprocessing (center crop and resize).
Applies the same transformations as used in ImageNet training.
"""

import os
import json
from typing import Optional, Callable, Tuple, List, Dict
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import center_crop_arr


class COCONpzDataset(Dataset):
    def __init__(self, path):
        self.data = np.load(path, mmap_mode="r")
        self.images = self.data["arr_0"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]).permute(2, 0, 1), None

class COCODataset(Dataset):
    """COCO dataset loader with ImageNet-style preprocessing.
    
    Applies center crop and resize transformations consistent with ImageNet preprocessing.
    
    Expected directory structure:
        root/
            train2017/
                000000000009.jpg
                000000000025.jpg
                ...
            val2017/
                000000000139.jpg
                000000000285.jpg
                ...
            annotations/
                instances_train2017.json
                instances_val2017.json
                captions_train2017.json (optional)
                captions_val2017.json (optional)
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        year: str = '2017',
        image_size: int = 256,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_captions: bool = False,
        filter_categories: Optional[List[int]] = None,
        return_raw_image: bool = False
    ):
        """
        Args:
            root: Root directory of COCO dataset
            split: Dataset split ('train' or 'val')
            year: COCO year ('2014' or '2017')
            image_size: Target size for center crop (default: 256)
            transform: Optional transform to be applied on images (applied after center crop)
            target_transform: Optional transform to be applied on labels
            use_captions: If True, load captions instead of instance annotations
            filter_categories: Optional list of category IDs to filter images by
            return_raw_image: If True, return original PIL image before preprocessing
        """
        self.root = root
        self.split = split
        self.year = year
        self.image_size = image_size
        self.transform = transform
        self.target_transform = target_transform
        self.use_captions = use_captions
        self.filter_categories = filter_categories
        self.return_raw_image = return_raw_image
        
        # Construct paths
        self.image_dir = os.path.join(root, f'{split}{year}')
        self.annot_dir = os.path.join(root, 'annotations')
        
        # Load annotations
        if use_captions:
            annot_file = os.path.join(self.annot_dir, f'captions_{split}{year}.json')
        else:
            annot_file = os.path.join(self.annot_dir, f'instances_{split}{year}.json')
        
        if not os.path.exists(annot_file):
            raise FileNotFoundError(f"Annotation file not found: {annot_file}")
        
        with open(annot_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image index
        self.images = {img['id']: img for img in self.coco_data['images']}
        
        # Build category index (only for instance annotations)
        if not use_captions:
            self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
            self.category_names = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Build annotations index
        self.image_to_annots = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_to_annots:
                self.image_to_annots[img_id] = []
            self.image_to_annots[img_id].append(ann)
        
        # Filter by categories if specified
        if filter_categories is not None and not use_captions:
            self.image_ids = [
                img_id for img_id in self.images.keys()
                if img_id in self.image_to_annots and
                any(ann['category_id'] in filter_categories 
                    for ann in self.image_to_annots[img_id])
            ]
        else:
            # Only keep images that have annotations
            self.image_ids = [
                img_id for img_id in self.images.keys()
                if img_id in self.image_to_annots
            ]
        
        self.image_ids = sorted(self.image_ids)
        
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Apply ImageNet-style center crop and resize preprocessing.
        
        Args:
            image: PIL Image in RGB format
            
        Returns:
            Preprocessed image as numpy array (H, W, 3) in [0, 255] uint8
        """
        # Convert to numpy array
        img_arr = np.array(image)
        
        # Apply center crop (same as ImageNet preprocessing)
        img_arr = center_crop_arr(img_arr, self.image_size)
        
        return img_arr
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            idx: Index
            
        Returns:
            tuple: (image, target) where:
                - image: Preprocessed image as torch.Tensor (C, H, W) in [0, 255] uint8
                - target: Dictionary with annotation information
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Store raw image if needed
        raw_image = image.copy() if self.return_raw_image else None
        
        # Apply center crop preprocessing (ImageNet-style)
        image_arr = self._preprocess_image(image)
        
        # Convert to tensor (C, H, W)
        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)
        
        # Apply optional transform (on top of center crop)
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        
        # Prepare target dictionary
        annotations = self.image_to_annots.get(img_id, [])
        
        if self.use_captions:
            # For captions
            target = {
                'image_id': img_id,
                'captions': [ann['caption'] for ann in annotations],
                'width': img_info['width'],
                'height': img_info['height'],
                'file_name': img_info['file_name']
            }
        else:
            # For instance annotations
            target = {
                'image_id': img_id,
                'annotations': annotations,
                'category_ids': [ann['category_id'] for ann in annotations],
                'width': img_info['width'],
                'height': img_info['height'],
                'file_name': img_info['file_name']
            }
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.return_raw_image:
            return image_tensor, target, raw_image
        
        return image_tensor, target
    
    def get_image_info(self, idx: int) -> Dict:
        """Get image metadata without loading the image."""
        img_id = self.image_ids[idx]
        return self.images[img_id]
    
    def get_category_name(self, category_id: int) -> str:
        """Get category name from category ID."""
        if self.use_captions:
            raise ValueError("Categories not available when use_captions=True")
        return self.category_names.get(category_id, "unknown")
    
    def get_all_categories(self) -> Dict[int, str]:
        """Get all category IDs and names."""
        if self.use_captions:
            raise ValueError("Categories not available when use_captions=True")
        return self.category_names.copy()


class SimpleCOCODataset(Dataset):
    """Simplified COCO dataset that just loads images from a directory.
    
    Similar to ImageNetDataset but for COCO structure. Applies same preprocessing.
    Does not require annotation files.
    
    Expected directory structure:
        root/
            train2017/
                *.jpg
            val2017/
                *.jpg
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        year: str = '2017',
        image_size: int = 256,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            root: Root directory of COCO dataset
            split: Dataset split ('train' or 'val')
            year: COCO year ('2014' or '2017')
            image_size: Target size for center crop (default: 256)
            transform: Optional transform to be applied on images (applied after center crop)
        """
        self.root = root
        self.split = split
        self.year = year
        self.image_size = image_size
        self.transform = transform
        
        # Construct image directory path
        self.image_dir = os.path.join(root, f'{split}{year}')
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Apply ImageNet-style center crop and resize preprocessing."""
        img_arr = np.array(image)
        img_arr = center_crop_arr(img_arr, self.image_size)
        return img_arr
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Args:
            idx: Index
            
        Returns:
            tuple: (image, filename) where:
                - image: Preprocessed image as torch.Tensor (C, H, W) in [0, 255] uint8
                - filename: Image filename
        """
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, filename)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # # Apply center crop preprocessing (ImageNet-style)
        # image_arr = self._preprocess_image(image)
        
        # # Convert to tensor (C, H, W)
        # image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)
        
        # Apply optional transform
        if self.transform is not None:
            image = self.transform(image)
        
        return image, filename


# Example usage
if __name__ == "__main__":

    # Example with simple dataset (no annotations)
    print("\nTesting SimpleCOCODataset without annotations...")
    simple_dataset = SimpleCOCODataset(
        root='/home/nsereyjo/iveco/datasets_iveco/COCO',
        split='train',
        year='2017',
        image_size=256
    )
    print(f"Simple dataset size: {len(simple_dataset)}")
    
    if len(simple_dataset) > 0:
        img, filename = simple_dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Filename: {filename}")
