#!/usr/bin/env python3
"""
Quick test script to verify preprocessing consistency using images from your dataset.
This script automatically finds test images and compares preprocessing methods.
"""

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Import your project modules
from utils import load_yaml_config
from data.datamodule import process_image_preserving_aspect_ratio


def quick_preprocessing_test(config_path="config.yaml", num_samples=3):
    """
    Quick test using images from the configured dataset
    """
    # Load configuration
    config = load_yaml_config(config_path)
    
    # Get dataset info
    dataset_name = config["data"]["dataset_name"]
    datasets_root = config["data"]["datasets_root"]
    
    # Find test images
    test_dir = os.path.join(datasets_root, dataset_name, "test", "images")
    if not os.path.exists(test_dir):
        # Try other splits
        for split in ["val", "train"]:
            test_dir = os.path.join(datasets_root, dataset_name, split, "images")
            if os.path.exists(test_dir):
                print(f"Using {split} split for testing (test split not found)")
                break
    
    if not os.path.exists(test_dir):
        print(f"Error: Could not find images in dataset {dataset_name}")
        return
    
    # Get image files
    image_files = glob.glob(os.path.join(test_dir, "*.png")) + \
                  glob.glob(os.path.join(test_dir, "*.jpg")) + \
                  glob.glob(os.path.join(test_dir, "*.jpeg"))
    
    if not image_files:
        print(f"No images found in {test_dir}")
        return
    
    # Limit to num_samples
    image_files = image_files[:num_samples]
    print(f"Testing {len(image_files)} images from {test_dir}")
    
    # Get preprocessing parameters
    preserve_aspect_ratio = config["data"].get("preserve_aspect_ratio", True)
    native_resolution = config["data"].get("native_resolution", False)
    
    if isinstance(config["data"]["img_size"], list):
        img_height, img_width = config["data"]["img_size"]
        target_size = (img_width, img_height)
    else:
        img_height = img_width = config["data"]["img_size"]
        target_size = (img_width, img_height)
    
    print(f"Configuration:")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Preserve aspect ratio: {preserve_aspect_ratio}")
    print(f"  - Native resolution: {native_resolution}")
    print(f"  - Target size: {target_size}")
    
    # Test each image
    all_identical = True
    
    for i, image_path in enumerate(image_files):
        print(f"\nTesting image {i+1}: {os.path.basename(image_path)}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        print(f"  Original size: {original_size}")
        
        # Method 1: Training-style processing
        if native_resolution:
            train_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            # Apply normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            train_tensor = (train_tensor - mean) / std
        elif preserve_aspect_ratio:
            train_tensor = process_image_preserving_aspect_ratio(
                image, target_size, is_mask=False, to_tensor=True, normalize=True
            )
        else:
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            train_tensor = transform(image)
        
        # Method 2: Inference-style processing (should be identical)
        # Import the inference function
        from inference import process_image_preserving_aspect_ratio as inf_process
        
        if native_resolution:
            inf_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            # Apply normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            inf_tensor = (inf_tensor - mean) / std
        elif preserve_aspect_ratio:
            inf_tensor = inf_process(
                image, target_size, is_mask=False, to_tensor=True, normalize=True
            )
        else:
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            inf_tensor = transform(image)
        
        # Compare
        identical = torch.allclose(train_tensor, inf_tensor, atol=1e-6)
        print(f"  Training shape: {train_tensor.shape}")
        print(f"  Inference shape: {inf_tensor.shape}")
        print(f"  Identical: {identical}")
        
        if not identical:
            diff = torch.abs(train_tensor - inf_tensor)
            print(f"  Max difference: {diff.max().item():.8f}")
            print(f"  Mean difference: {diff.mean().item():.8f}")
            all_identical = False
        
        # Quick visualization for first image
        if i == 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original
            axes[0].imshow(image)
            axes[0].set_title(f'Original\n{original_size}')
            axes[0].axis('off')
            
            # Training processed (denormalized)
            train_denorm = train_tensor.clone()
            for t, m, s in zip(train_denorm, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
                t.mul_(s).add_(m)
            train_denorm = torch.clamp(train_denorm, 0, 1)
            
            axes[1].imshow(train_denorm.permute(1, 2, 0))
            axes[1].set_title(f'Training Pipeline\n{train_tensor.shape}')
            axes[1].axis('off')
            
            # Inference processed (denormalized)
            inf_denorm = inf_tensor.clone()
            for t, m, s in zip(inf_denorm, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
                t.mul_(s).add_(m)
            inf_denorm = torch.clamp(inf_denorm, 0, 1)
            
            axes[2].imshow(inf_denorm.permute(1, 2, 0))
            axes[2].set_title(f'Inference Pipeline\n{inf_tensor.shape}')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"quick_test_{os.path.basename(image_path)}.png", dpi=150)
            print(f"  Saved visualization: quick_test_{os.path.basename(image_path)}.png")
            plt.show()
    
    # Final summary
    print("\n" + "="*50)
    print("QUICK TEST SUMMARY")
    print("="*50)
    print(f"Dataset: {dataset_name}")
    print(f"Images tested: {len(image_files)}")
    print(f"All preprocessing identical: {all_identical}")
    
    if all_identical:
        print("✅ SUCCESS: Training and inference preprocessing are consistent!")
    else:
        print("⚠️  WARNING: Found differences in preprocessing!")
    
    print("="*50)
    
    return all_identical


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick preprocessing consistency test")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--samples", type=int, default=3, help="Number of sample images to test")
    
    args = parser.parse_args()
    
    success = quick_preprocessing_test(args.config, args.samples)
    
    if not success:
        exit(1)