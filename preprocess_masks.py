#!/usr/bin/env python3
"""
Script to preprocess masks to ensure they are strictly binary.
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

def binarize_mask(mask_path, output_path=None, threshold=0):
    """
    Convert mask to strict binary (0 and 255).
    Args:
        mask_path: Path to the input mask
        output_path: Path to save processed mask (if None, overwrites original)
        threshold: Pixel values above this will be set to 255, below to 0
    """
    # Read mask
    mask = np.array(Image.open(mask_path).convert('L'))
    
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    
    # Save binary mask
    output_path = output_path or mask_path
    Image.fromarray(binary_mask).save(output_path)
    
    return binary_mask

def process_dataset(dataset_dir, threshold=0, save_originals=True):
    """
    Process all masks in a dataset directory.
    """
    # Find all mask directories
    mask_dirs = []
    for split in ['train', 'val', 'test']:
        mask_dir = os.path.join(dataset_dir, split, 'masks')
        if os.path.exists(mask_dir):
            mask_dirs.append(mask_dir)
    
    total_processed = 0
    for mask_dir in mask_dirs:
        print(f"Processing masks in {mask_dir}")
        
        # Create backup directory if needed
        if save_originals:
            backup_dir = os.path.join(os.path.dirname(mask_dir), 'masks_original')
            os.makedirs(backup_dir, exist_ok=True)
        
        # Process all masks
        mask_files = glob.glob(os.path.join(mask_dir, '*.png'))
        for mask_path in tqdm(mask_files):
            filename = os.path.basename(mask_path)
            
            # Backup original if needed
            if save_originals:
                backup_path = os.path.join(backup_dir, filename)
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(mask_path, backup_path)
            
            # Binarize mask
            binarize_mask(mask_path, threshold=threshold)
            total_processed += 1
    
    print(f"Processed {total_processed} masks. All masks are now strictly binary.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess masks to ensure binary values")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--threshold", type=int, default=0, help="Threshold for binarization")
    parser.add_argument("--no-backup", action="store_true", help="Don't backup original masks")
    args = parser.parse_args()
    
    process_dataset(args.dataset_dir, args.threshold, not args.no_backup)