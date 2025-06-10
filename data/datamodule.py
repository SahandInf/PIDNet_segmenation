import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import glob
import math


def process_image_preserving_aspect_ratio(image, target_size, is_mask=False, to_tensor=True, normalize=True):
    """
    Process an image while preserving aspect ratio, for both training and inference.
    
    Args:
        image: PIL Image to process
        target_size: Tuple of (width, height) for the target size
        is_mask: Whether this image is a mask (affects interpolation mode)
        to_tensor: Whether to convert to tensor
        normalize: Whether to normalize (only for non-mask images)
        
    Returns:
        Processed image (as tensor if to_tensor=True, else PIL Image)
    """
    # Get current dimensions
    width, height = image.size
    aspect_ratio = width / height
    
    # Calculate new dimensions preserving aspect ratio
    target_w, target_h = target_size
    if aspect_ratio > target_w / target_h:  # If image is wider than target
        new_width = target_w
        new_height = int(new_width / aspect_ratio)
    else:  # If image is taller than target
        new_height = target_h
        new_width = int(new_height * aspect_ratio)
    
    # Resize with appropriate interpolation method
    interpolation = Image.NEAREST if is_mask else Image.BILINEAR
    resized_img = image.resize((new_width, new_height), interpolation)
    
    # Create padding
    pad_w = max(0, target_w - new_width)
    pad_h = max(0, target_h - new_height)
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)  # left, top, right, bottom
    
    # Apply padding
    padded_img = Image.new(image.mode, target_size, 0)
    padded_img.paste(resized_img, (padding[0], padding[1]))
    
    if to_tensor:
        # Convert to tensor
        if is_mask:
            # For masks, just convert to tensor
            return transforms.ToTensor()(padded_img)
        else:
            # For images, apply normalization if requested
            image_tensor = transforms.ToTensor()(padded_img)
            if normalize:
                image_tensor = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )(image_tensor)
            return image_tensor
    else:
        return padded_img


class SegmentationDataset(Dataset):
    """Dataset class for image segmentation with aspect ratio preservation."""
    
    def __init__(self, img_dir, transform=None, target_transform=None, img_suffix='*.png', mask_suffix='*.png', 
                 preserve_aspect_ratio=True, target_size=(160, 416), native_resolution=False):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.target_size = target_size
        self.native_resolution = native_resolution
        
        # Find all image and mask files
        self.img_files = sorted(glob.glob(os.path.join(img_dir, 'images', img_suffix)))
        self.mask_files = sorted(glob.glob(os.path.join(img_dir, 'masks', mask_suffix)))
        
        if len(self.img_files) != len(self.mask_files):
            raise ValueError(f"Number of images ({len(self.img_files)}) doesn't match number of masks ({len(self.mask_files)})")
        
        # Print info about the dataset
        if len(self.img_files) > 0:
            # Get sample image size
            sample_img = Image.open(self.img_files[0])
            self.original_size = sample_img.size
            print(f"Dataset contains {len(self.img_files)} images with original size: {self.original_size}")
            if native_resolution:
                print(f"Using native resolution: {self.original_size}")
            elif preserve_aspect_ratio:
                print(f"Preserving aspect ratio with target size: {target_size}")
            else:
                print(f"Resizing to: {target_size}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = self.mask_files[idx]
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        if self.native_resolution:
            # Use original image size without resizing
            if self.transform:
                # Only apply normalization and conversion to tensor
                image = transforms.ToTensor()(image)
                image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            else:
                image = transforms.ToTensor()(image)
            
            if self.target_transform:
                # Only apply conversion to tensor for mask
                mask = transforms.ToTensor()(mask)
            else:
                mask = transforms.ToTensor()(mask)
        
        elif self.preserve_aspect_ratio:
            # Use the unified aspect ratio preservation function
            # FIXED: Use target_size directly as it's already in (width, height) format
            image = process_image_preserving_aspect_ratio(
                image, 
                self.target_size,  # FIXED: Use directly, no swapping
                is_mask=False, 
                to_tensor=True, 
                normalize=True
            )
            
            mask = process_image_preserving_aspect_ratio(
                mask, 
                self.target_size,  # FIXED: Use directly, no swapping
                is_mask=True, 
                to_tensor=True, 
                normalize=False
            )
            
        else:
            # Apply standard transformations (resize without preserving aspect ratio)
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform if none provided
                image = transforms.Compose([
                    transforms.Resize(self.target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(image)
            
            if self.target_transform:
                mask = self.target_transform(mask)
            else:
                # Default transform if none provided
                mask = transforms.Compose([
                    transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.ToTensor()
                ])(mask)
        
        return image, mask


class SegmentationDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for segmentation with aspect ratio preservation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["training"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.pin_memory = config["data"]["pin_memory"]
        
        # Get image size settings from config
        self.preserve_aspect_ratio = config["data"].get("preserve_aspect_ratio", True)
        self.native_resolution = config["data"].get("native_resolution", False)
        
        # FIXED: Determine target image size consistently with inference.py
        if self.native_resolution:
            # Will use original image size
            self.target_size = None
        elif "img_size" in config["data"]:
            if isinstance(config["data"]["img_size"], list) or isinstance(config["data"]["img_size"], tuple):
                # FIXED: Consistent interpretation with inference.py
                img_height, img_width = config["data"]["img_size"]
                self.target_size = (img_width, img_height)  # PIL format (width, height)
                print(f"Datamodule target_size: {self.target_size} from config {config['data']['img_size']}")
            else:
                # If img_size is a single value, convert to (width, height)
                size = config["data"]["img_size"]
                self.target_size = (size, size)
        else:
            # Default to dimensions that are multiples of 32 close to 160x401
            self.target_size = (416, 160)  # (width, height)
        
        # Define the dataset and directories
        self.dataset_name = config["data"]["dataset_name"]
        self.datasets_root = config["data"]["datasets_root"]
        
        # Construct the full paths for the selected dataset
        dataset_path = os.path.join(self.datasets_root, self.dataset_name)
        self.train_dir = os.path.join(dataset_path, "train")
        self.val_dir = os.path.join(dataset_path, "val")
        self.test_dir = os.path.join(dataset_path, "test")
        
        # Define transformations
        self.setup_transforms()
        
        print(f"Using dataset: {self.dataset_name}")
        print(f"Train data: {self.train_dir}")
        print(f"Validation data: {self.val_dir}")
        print(f"Test data: {self.test_dir}")
        
        # Print image size info
        if self.native_resolution:
            print(f"Using native image resolution")
        elif self.preserve_aspect_ratio:
            print(f"Preserving aspect ratio with target size: {self.target_size}")
        else:
            print(f"Resizing images to: {self.target_size}")

    def setup_transforms(self):
        # If using native resolution or preserving aspect ratio, we'll handle resize and padding
        # in the dataset __getitem__ method
        if self.native_resolution or self.preserve_aspect_ratio:
            # Only define normalization
            self.img_transform = None  # Will handle in __getitem__
            self.mask_transform = None  # Will handle in __getitem__
        else:
            # Standard transform with resize
            self.img_transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Define mask transformations
            self.mask_transform = transforms.Compose([
                transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])

    def setup(self, stage=None):
        # Create datasets for training, validation, and testing
        if stage == 'fit' or stage is None:
            self.train_dataset = SegmentationDataset(
                self.train_dir,
                transform=self.img_transform,
                target_transform=self.mask_transform,
                mask_suffix='*.png',
                preserve_aspect_ratio=self.preserve_aspect_ratio,
                target_size=self.target_size,
                native_resolution=self.native_resolution
            )
            
            self.val_dataset = SegmentationDataset(
                self.val_dir,
                transform=self.img_transform,
                target_transform=self.mask_transform,
                mask_suffix='*.png',
                preserve_aspect_ratio=self.preserve_aspect_ratio,
                target_size=self.target_size,
                native_resolution=self.native_resolution
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = SegmentationDataset(
                self.test_dir,
                transform=self.img_transform,
                target_transform=self.mask_transform,
                mask_suffix='*.png',
                preserve_aspect_ratio=self.preserve_aspect_ratio,
                target_size=self.target_size,
                native_resolution=self.native_resolution
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )