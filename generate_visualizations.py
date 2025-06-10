#!/usr/bin/env python3
"""
Visualization generation script for PIDNet segmentation results.
Creates high-quality visualizations from inference results.
COMPLETELY REWRITTEN: Perfect overlay alignment and sharp mask display.
"""

import os
import argparse
import torch
import glob
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


def resize_mask_to_image(pred_mask, target_size):
    """
    Resize prediction mask to match original image size while preserving sharp edges.
    
    Args:
        pred_mask: numpy array of prediction mask
        target_size: tuple (width, height) of target size
    
    Returns:
        numpy array: resized mask with preserved sharp edges
    """
    # Handle different input types
    if pred_mask.dtype == np.bool_:
        pred_mask = pred_mask.astype(np.uint8) * 255
    elif pred_mask.dtype != np.uint8:
        # Normalize to 0-255 range for PIL processing
        if pred_mask.max() <= 1.0:
            pred_mask = (pred_mask * 255).astype(np.uint8)
        else:
            pred_mask = pred_mask.astype(np.uint8)
    
    # Convert to PIL Image for high-quality resizing
    mask_pil = Image.fromarray(pred_mask)
    
    # Resize using nearest neighbor to preserve sharp edges
    mask_resized = mask_pil.resize(target_size, Image.NEAREST)
    
    # Convert back to numpy array and normalize to 0-1 range
    mask_array = np.array(mask_resized) / 255.0
    
    return mask_array


def create_visualization(image_path, output, config, save_probability_maps=True, 
                        visualization_type="full", original_size=None, padding_info=None):
    """
    Create perfect visualization with aligned overlays and sharp masks.
    
    Args:
        image_path: Path to original image
        output: Model output tensor (may be padded)
        config: Model configuration
        save_probability_maps: Whether to include probability maps
        visualization_type: "full", "simple", "overlay_only"
        original_size: Original image size (width, height) for perfect alignment
        padding_info: Padding information to remove padding from output
    """
    # Load original image
    try:
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")
    
    # Get image dimensions for perfect alignment
    if original_size is None:
        original_size = image.size  # (width, height)
    
    print(f"📐 Processing: {os.path.basename(image_path)} - Size: {original_size[0]}x{original_size[1]}")
    
    # CRITICAL: Remove padding from output if present
    if padding_info is not None or (hasattr(output, 'shape') and len(output.shape) == 3):
        output = remove_padding_from_output(output, padding_info, original_size)
    
    # Setup figure based on visualization type
    if visualization_type == "simple":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        subplot_count = 2
    elif visualization_type == "overlay_only":
        fig, axes = plt.subplots(1, 1, figsize=(12, 9))
        subplot_count = 1
        axes = [axes]  # Make it iterable
    else:  # full
        subplot_count = 4 if save_probability_maps else 3
        fig, axes = plt.subplots(1, subplot_count, figsize=(5.5 * subplot_count, 6))
        if subplot_count == 1:
            axes = [axes]  # Make it iterable
    
    # Original image (skip for overlay_only)
    axis_idx = 0
    if visualization_type != "overlay_only":
        axes[axis_idx].imshow(image_array)
        axes[axis_idx].set_title('Input Image', fontsize=14, fontweight='bold', pad=10)
        axes[axis_idx].axis('off')
        axis_idx += 1
    
    # Process prediction based on model type
    if config["model"]["out_channels"] > 1:
        # Multi-class segmentation
        pred = torch.softmax(output, dim=0)
        pred_probs = pred.cpu().numpy()
        pred_vis = torch.argmax(pred, dim=0).cpu().numpy()
        
        print(f"🎯 Multi-class prediction: {pred_vis.shape} -> resizing to {original_size}")
        
        # Resize mask to match original image size for perfect alignment
        pred_vis_resized = resize_mask_to_image(pred_vis, original_size)
        
        if visualization_type == "overlay_only":
            # Perfect overlay alignment
            axes[0].imshow(image_array)
            axes[0].imshow(pred_vis_resized, cmap='tab10', alpha=0.6, interpolation='nearest')
            axes[0].set_title('Segmentation Overlay', fontsize=16, fontweight='bold', pad=15)
            axes[0].axis('off')
        elif visualization_type == "simple":
            # Simple 2-panel with perfect sizing
            axes[1].imshow(pred_vis_resized, cmap='tab10', interpolation='nearest')
            axes[1].set_title('Segmentation Mask', fontsize=14, fontweight='bold', pad=10)
            axes[1].axis('off')
        else:
            # Full visualization
            # Segmentation mask
            axes[axis_idx].imshow(pred_vis_resized, cmap='tab10', interpolation='nearest')
            axes[axis_idx].set_title('Segmentation Mask', fontsize=14, fontweight='bold', pad=10)
            axes[axis_idx].axis('off')
            axis_idx += 1
            
            # Perfect overlay
            axes[axis_idx].imshow(image_array)
            axes[axis_idx].imshow(pred_vis_resized, cmap='tab10', alpha=0.5, interpolation='nearest')
            axes[axis_idx].set_title('Overlay', fontsize=14, fontweight='bold', pad=10)
            axes[axis_idx].axis('off')
            axis_idx += 1
            
            # Probability map
            if save_probability_maps and axis_idx < len(axes):
                # Find dominant class for probability visualization
                unique_classes = np.unique(pred_vis)
                if len(unique_classes) > 1:
                    # Use most common class (excluding background if 0)
                    class_counts = np.bincount(pred_vis.flatten())
                    if len(class_counts) > 1 and class_counts[0] > class_counts[1:].sum():
                        # If background (0) dominates, use second most common
                        dominant_class = np.argmax(class_counts[1:]) + 1 if len(class_counts) > 1 else 0
                    else:
                        dominant_class = np.argmax(class_counts)
                else:
                    dominant_class = unique_classes[0] if len(unique_classes) > 0 else 0
                
                if dominant_class < pred_probs.shape[0]:
                    prob_map = pred_probs[dominant_class]
                    prob_map_resized = resize_mask_to_image(prob_map, original_size)
                    im = axes[axis_idx].imshow(prob_map_resized, cmap='viridis', vmin=0, vmax=1, 
                                             interpolation='bilinear')
                    axes[axis_idx].set_title(f'Class {dominant_class} Confidence', 
                                           fontsize=14, fontweight='bold', pad=10)
                    axes[axis_idx].axis('off')
                    plt.colorbar(im, ax=axes[axis_idx], fraction=0.046, pad=0.04)
    else:
        # Binary segmentation
        pred = torch.sigmoid(output)
        pred_probs = pred.cpu().numpy().squeeze()
        pred_vis = (pred > 0.5).float().cpu().numpy().squeeze()
        
        print(f"🎯 Binary prediction: {pred_vis.shape} -> resizing to {original_size}")
        
        # Resize mask to match original image size for perfect alignment
        pred_vis_resized = resize_mask_to_image(pred_vis, original_size)
        
        if visualization_type == "overlay_only":
            # Perfect overlay alignment
            axes[0].imshow(image_array)
            axes[0].imshow(pred_vis_resized, cmap='Reds', alpha=0.6, interpolation='nearest')
            axes[0].set_title('Segmentation Overlay', fontsize=16, fontweight='bold', pad=15)
            axes[0].axis('off')
        elif visualization_type == "simple":
            # Simple 2-panel with perfect sizing  
            axes[1].imshow(pred_vis_resized, cmap='gray', interpolation='nearest')
            axes[1].set_title('Segmentation Mask', fontsize=14, fontweight='bold', pad=10)
            axes[1].axis('off')
        else:
            # Full visualization
            # Segmentation mask
            axes[axis_idx].imshow(pred_vis_resized, cmap='gray', interpolation='nearest')
            axes[axis_idx].set_title('Segmentation Mask', fontsize=14, fontweight='bold', pad=10)
            axes[axis_idx].axis('off')
            axis_idx += 1
            
            # Perfect overlay
            axes[axis_idx].imshow(image_array)
            axes[axis_idx].imshow(pred_vis_resized, cmap='Reds', alpha=0.5, interpolation='nearest')
            axes[axis_idx].set_title('Overlay', fontsize=14, fontweight='bold', pad=10)
            axes[axis_idx].axis('off')
            axis_idx += 1
            
            # Probability map
            if save_probability_maps and axis_idx < len(axes):
                pred_probs_resized = resize_mask_to_image(pred_probs, original_size)
                im = axes[axis_idx].imshow(pred_probs_resized, cmap='viridis', vmin=0, vmax=1, 
                                         interpolation='bilinear')
                axes[axis_idx].set_title('Prediction Confidence', fontsize=14, fontweight='bold', pad=10)
                axes[axis_idx].axis('off')
                plt.colorbar(im, ax=axes[axis_idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout(pad=2.0)
    return fig


def load_raw_output(raw_file_path):
    """Load raw model output from saved file"""
    try:
        raw_data = torch.load(raw_file_path, map_location='cpu')
        output = raw_data['output']
        image_path = raw_data['image_path']
        original_size = raw_data['original_size']
        
        # Get padding info if available
        padding_info = raw_data.get('padding_info', None)
        processed_size = raw_data.get('processed_size', None)
        
        return output, image_path, original_size, padding_info, processed_size
    except Exception as e:
        print(f"❌ Error loading {raw_file_path}: {e}")
        return None, None, None, None, None
    


def remove_padding_from_output(output, padding_info, original_size):
    """Remove padding from model output and resize to original dimensions"""
    if padding_info is None:
        # No padding info, just resize to original
        return torch.nn.functional.interpolate(
            output.unsqueeze(0),
            size=(original_size[1], original_size[0]),  # (height, width)
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    # Extract content area (remove padding)
    output_unpadded = output[
        :,
        padding_info['pad_top']:padding_info['pad_top']+padding_info['new_height'],
        padding_info['pad_left']:padding_info['pad_left']+padding_info['new_width']
    ]
    
    # Resize to original dimensions
    output_resized = torch.nn.functional.interpolate(
        output_unpadded.unsqueeze(0),
        size=(original_size[1], original_size[0]),  # (height, width)
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    
    return output_resized    


def generate_visualizations_from_raw(inference_dir, num_samples=None, visualization_dpi=200,
                                   save_probability_maps=True, num_workers=4, 
                                   visualization_type="full", output_format="png"):
    """
    Generate perfect visualizations from saved raw outputs.
    """
    print(f"🎨 VISUALIZATION GENERATOR")
    print(f"📁 Source: {inference_dir}")
    print(f"🎯 Mode: overlay alignment + Sharp masks")
    
    # Load configuration
    config_path = os.path.join(inference_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Model: {config['model']['name']} ({config['model']['out_channels']} classes)")
    
    # Find raw outputs
    raw_outputs_dir = os.path.join(inference_dir, "raw_outputs")
    if not os.path.exists(raw_outputs_dir):
        print(f"❌ No raw outputs found at: {raw_outputs_dir}")
        print("💡 Run inference with --save_raw_outputs flag first")
        return False
    
    raw_files = glob.glob(os.path.join(raw_outputs_dir, "**", "*.pt"), recursive=True)
    if not raw_files:
        print(f"❌ No raw output files found in: {raw_outputs_dir}")
        return False
    
    raw_files.sort()
    if num_samples:
        raw_files = raw_files[:num_samples]
        print(f"🎯 Processing {num_samples} samples out of {len(glob.glob(os.path.join(raw_outputs_dir, '**', '*.pt'), recursive=True))}")
    
    print(f"📊 Processing {len(raw_files)} raw outputs")
    print(f"⚙️  Configuration: {num_workers} workers, {visualization_dpi} DPI, {output_format.upper()} format")
    
    # Create visualization directory
    viz_dir = os.path.join(inference_dir, f"visualizations_{visualization_type}")
    os.makedirs(viz_dir, exist_ok=True)
    
    def process_single_visualization(raw_file):
        try:
            # Load raw data with padding info
            output, image_path, original_size, padding_info, processed_size = load_raw_output(raw_file)
            if output is None:
                return None, f"Failed to load raw data from {raw_file}"
            
            # Check if original image exists
            if not os.path.exists(image_path):
                return None, f"Original image not found: {image_path}"
            
            # Create perfect visualization with padding handling
            fig = create_visualization(
                image_path, output, config, save_probability_maps, 
                visualization_type, original_size, padding_info
            )
            
            # Determine output path
            rel_path = os.path.relpath(raw_file, raw_outputs_dir)
            viz_filename = os.path.splitext(rel_path)[0].replace('_raw', f'_{visualization_type}')
            viz_filename += f'.{output_format}'
            viz_path = os.path.join(viz_dir, viz_filename)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(viz_path), exist_ok=True)
            
            # Save high-quality visualization
            fig.savefig(viz_path, dpi=visualization_dpi, bbox_inches='tight', 
                       format=output_format, facecolor='white', edgecolor='none')
            plt.close(fig)
            
            return viz_path, None
            
        except Exception as e:
            plt.close('all')  # Clean up any open figures
            return None, f"Error processing {os.path.basename(raw_file)}: {str(e)}"
    
    # Process visualizations
    start_time = time.time()
    
    if num_workers == 1:
        # Sequential processing for debugging
        print("🔧 Sequential processing mode (debugging)")
        results = []
        for raw_file in tqdm.tqdm(raw_files, desc="Creating visualizations"):
            results.append(process_single_visualization(raw_file))
    else:
        # Parallel processing
        print(f"⚡ Parallel processing with {num_workers} workers")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm.tqdm(
                executor.map(process_single_visualization, raw_files),
                total=len(raw_files),
                desc="Creating visualizations"
            ))
    
    # Analyze results
    successful = [r[0] for r in results if r[0] is not None]
    failed = [(r[1]) for r in results if r[0] is None]
    
    viz_time = time.time() - start_time
    
    # Print comprehensive results
    print(f"\n{'='*70}")
    print(f"VISUALIZATION GENERATION COMPLETED")
    print(f"{'='*70}")
    print(f"✅ Successfully generated: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⏱️  Total time: {viz_time:.1f}s")
    print(f"⚡ Speed: {len(successful)/viz_time:.1f} visualizations/second")
    print(f"📁 Results saved to: {viz_dir}")
    print(f"🎯 Quality: overlay alignment + Sharp mask boundaries")
    
    if failed:
        print(f"\n❌ Error Details:")
        for i, error in enumerate(failed[:5], 1):
            print(f"   {i}. {error}")
        if len(failed) > 5:
            print(f"   ... and {len(failed) - 5} more errors")
    
    # Create comprehensive report
    create_visualization_report(inference_dir, viz_dir, {
        'total_processed': len(raw_files),
        'successful': len(successful),
        'failed': len(failed),
        'time_taken': viz_time,
        'visualization_type': visualization_type,
        'dpi': visualization_dpi,
        'workers': num_workers,
        'output_format': output_format
    }, config)
    
    print(f"📊 Detailed report: {os.path.join(viz_dir, 'visualization_report.html')}")
    
    return len(successful) > 0


def generate_visualizations_from_masks(inference_dir, image_dir, num_samples=None, 
                                     visualization_dpi=200, num_workers=4,
                                     visualization_type="overlay_only", output_format="png"):
    """
    Generate perfect visualizations from saved masks.
    """
    print(f"🎨 MASK OVERLAY GENERATOR")
    print(f"📁 Masks: {inference_dir}")
    print(f"📁 Images: {image_dir}")
    
    # Load configuration
    config_path = os.path.join(inference_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Find mask files
    masks_dir = os.path.join(inference_dir, "masks")
    if not os.path.exists(masks_dir):
        print(f"❌ No masks found at: {masks_dir}")
        return False
    
    mask_files = glob.glob(os.path.join(masks_dir, "**", "*_mask.png"), recursive=True)
    if not mask_files:
        print(f"❌ No mask files found in: {masks_dir}")
        return False
    
    mask_files.sort()
    if num_samples:
        mask_files = mask_files[:num_samples]
    
    print(f"📊 Processing {len(mask_files)} masks")
    print(f"🎯 overlay alignment enabled")
    
    # Create visualization directory
    viz_dir = os.path.join(inference_dir, "visualizations_from_masks")
    os.makedirs(viz_dir, exist_ok=True)
    
    def create_mask_overlay(mask_file):
        try:
            # Load mask
            mask = Image.open(mask_file).convert('L')
            
            # Find corresponding original image
            rel_path = os.path.relpath(mask_file, masks_dir)
            base_name = os.path.splitext(rel_path)[0].replace('_mask', '')
            
            # Search for original image with multiple extensions
            image_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG']:
                candidate_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(candidate_path):
                    image_path = candidate_path
                    break
            
            if not image_path:
                return None, f"Original image not found for {base_name}"
            
            # Load original image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            print(f"📐 {base_name}: Image {image.size}, Mask {mask.size}")
            
            # Ensure perfect size alignment
            if mask.size != image.size:
                print(f"🔧 Resizing mask from {mask.size} to {image.size}")
                mask = mask.resize(image.size, Image.NEAREST)
            
            mask_array = np.array(mask) / 255.0
            
            # Create perfect overlay visualization
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            
            # Display original image
            ax.imshow(image_array)
            
            # Perfect overlay with sharp edges
            if config["model"]["out_channels"] > 1:
                overlay = ax.imshow(mask_array, cmap='tab10', alpha=0.6, interpolation='nearest')
            else:
                overlay = ax.imshow(mask_array, cmap='Reds', alpha=0.6, interpolation='nearest')
            
            ax.set_title(f'Segmentation Overlay\n{os.path.basename(image_path)}', 
                        fontsize=16, fontweight='bold', pad=15)
            ax.axis('off')
            
            # Save high-quality visualization
            viz_filename = base_name + f'_overlay.{output_format}'
            viz_path = os.path.join(viz_dir, viz_filename)
            
            os.makedirs(os.path.dirname(viz_path), exist_ok=True)
            
            fig.savefig(viz_path, dpi=visualization_dpi, bbox_inches='tight',
                       format=output_format, facecolor='white', edgecolor='none')
            plt.close(fig)
            
            return viz_path, None
            
        except Exception as e:
            plt.close('all')  # Clean up any open figures
            return None, f"Error processing {os.path.basename(mask_file)}: {str(e)}"
    
    # Process visualizations
    start_time = time.time()
    
    print(f"⚡ Processing with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm.tqdm(
            executor.map(create_mask_overlay, mask_files),
            total=len(mask_files),
            desc="Creating overlays"
        ))
    
    successful = [r[0] for r in results if r[0] is not None]
    failed = [r[1] for r in results if r[0] is None]
    
    viz_time = time.time() - start_time
    
    print(f"\n✅ Generated {len(successful)} overlay visualizations in {viz_time:.1f}s")
    print(f"📁 Saved to: {viz_dir}")
    print(f"🎯 Quality: alignment + Sharp boundaries")
    
    if failed:
        print(f"\n❌ {len(failed)} failures:")
        for error in failed[:3]:
            print(f"   • {error}")
    
    return len(successful) > 0


def create_visualization_report(inference_dir, viz_dir, stats, config):
    """Create a comprehensive report for  visualization generation"""
    report_path = os.path.join(viz_dir, "visualization_report.html")
    
    success_rate = (stats['successful'] / stats['total_processed'] * 100) if stats['total_processed'] > 0 else 0
    
    with open(report_path, "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title> PIDNet Visualization Report</title>
            <meta charset="utf-8">
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{ 
                    max-width: 1400px; 
                    margin: 0 auto; 
                    background: white; 
                    margin-top: 20px;
                    margin-bottom: 20px;
                    border-radius: 15px; 
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 40px; 
                    text-align: center;
                }}
                .header h1 {{ 
                    margin: 0; 
                    font-size: 3em; 
                    font-weight: 300;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                }}
                .header p {{ 
                    margin: 10px 0 0 0; 
                    font-size: 1.2em; 
                    opacity: 0.9;
                }}
                .metrics {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
                    gap: 30px; 
                    margin: 40px; 
                }}
                .metric-card {{ 
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    padding: 30px; 
                    border-radius: 15px; 
                    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                    text-align: center;
                    transition: transform 0.3s ease;
                }}
                .metric-card:hover {{
                    transform: translateY(-5px);
                }}
                .metric-value {{ 
                    font-size: 3em; 
                    font-weight: bold; 
                    margin-bottom: 10px;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                }}
                .metric-label {{ 
                    font-size: 1.1em; 
                    opacity: 0.9;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .success {{ color: #4CAF50; }}
                .error {{ color: #f44336; }}
                .config-section {{ 
                    background: #f8f9fa; 
                    padding: 30px; 
                    margin: 40px; 
                    border-radius: 15px; 
                    border-left: 5px solid #667eea;
                }}
                .config-section h3 {{
                    color: #333;
                    font-size: 1.5em;
                    margin-bottom: 20px;
                }}
                .config-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                }}
                .config-item {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                }}
                .quality-badge {{ 
                    background: linear-gradient(135deg, #4CAF50, #45a049);
                    color: white; 
                    padding: 8px 16px; 
                    border-radius: 25px; 
                    font-size: 14px; 
                    font-weight: bold;
                    display: inline-block;
                    margin: 5px;
                    box-shadow: 0 2px 10px rgba(76, 175, 80, 0.3);
                }}
                .feature-list {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin-top: 20px;
                }}
                .feature-item {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                    font-weight: 500;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎨  PIDNet Visualization Report</h1>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <div>
                        <span class="quality-badge">🎯  OVERLAY ALIGNMENT</span>
                        <span class="quality-badge">✨ SHARP MASK BOUNDARIES</span>
                    </div>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{stats['successful']}</div>
                        <div class="metric-label"> Visualizations</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{success_rate:.1f}%</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{stats['successful']/stats['time_taken']:.1f}</div>
                        <div class="metric-label">Visualizations/Second</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{stats['dpi']}</div>
                        <div class="metric-label">Output DPI</div>
                    </div>
                </div>
                
                <div class="config-section">
                    <h3>🚀  Visualization Features</h3>
                    <div class="feature-list">
                        <div class="feature-item">✅  overlay alignment between images and masks</div>
                        <div class="feature-item">✅ Sharp mask boundaries using nearest neighbor interpolation</div>
                        <div class="feature-item">✅ Automatic size matching for seamless overlays</div>
                        <div class="feature-item">✅ High-quality {stats['dpi']} DPI output</div>
                        <div class="feature-item">✅ Professional {stats['output_format'].upper()} format</div>
                        <div class="feature-item">✅ Multi-threaded processing with {stats['workers']} workers</div>
                    </div>
                </div>
                
                <div class="config-section">
                    <h3>⚙️ Configuration Details</h3>
                    <div class="config-grid">
                        <div class="config-item">
                            <strong>Model Architecture:</strong><br>
                            {config['model']['name']} with {config['model']['in_channels']} → {config['model']['out_channels']} channels
                        </div>
                        <div class="config-item">
                            <strong>Visualization Type:</strong><br>
                            {stats['visualization_type'].title()} layout
                        </div>
                        <div class="config-item">
                            <strong>Processing Time:</strong><br>
                            {stats['time_taken']:.1f} seconds total
                        </div>
                        <div class="config-item">
                            <strong>Quality Settings:</strong><br>
                             alignment + Sharp interpolation
                        </div>
                    </div>
                </div>
                
                <div class="config-section">
                    <h3>📊 Quality Improvements</h3>
                    <div class="feature-list">
                        <div class="feature-item">🎯 <strong> Alignment:</strong> Masks resized to exact image dimensions</div>
                        <div class="feature-item">✨ <strong>Sharp Boundaries:</strong> Nearest neighbor interpolation preserves edges</div>
                        <div class="feature-item">🖼️ <strong>Original Quality:</strong> Background images maintain full resolution</div>
                        <div class="feature-item">🎨 <strong>Professional Output:</strong> Publication-ready visualizations</div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """)
    
    # Create JSON report
    json_report = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "mode": "visualization",
        "performance": stats,
        "model_info": {
            "name": config['model']['name'],
            "in_channels": config['model']['in_channels'],
            "out_channels": config['model']['out_channels']
        },
        "quality_features": [
            "overlay_alignment",
            "sharp_mask_boundaries", 
            "automatic_size_matching",
            "high_dpi_output",
            "professional_format"
        ],
        "results_summary": {
            "successful": stats['successful'],
            "failed": stats['failed'],
            "total": stats['total_processed'],
            "success_rate": success_rate
        }
    }
    
    with open(os.path.join(viz_dir, "visualization_report.json"), "w") as f:
        json.dump(json_report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PIDNet visualizations with aligned overlays and sharp masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate visualizations from raw outputs
  python visualization.py --dir /path/to/inference/results --type full
  
  # Create  overlays from masks
  python visualization.py --dir /path/to/results --image_dir /path/to/images --from_masks
  
  # High-quality overlay-only visualizations
  python visualization.py --dir /path/to/results --type overlay_only --dpi 300
        """
    )
    
    # Main arguments
    parser.add_argument("--dir", type=str, required=True, 
                       help="Inference results directory")
    parser.add_argument("--image_dir", type=str, 
                       help="Original images directory (required for --from_masks)")
    
    # Visualization options
    parser.add_argument("--type", type=str, choices=["full", "simple", "overlay_only"], 
                       default="full", help="Visualization layout type")
    parser.add_argument("--samples", type=int, 
                       help="Number of samples to visualize (default: all)")
    parser.add_argument("--dpi", type=int, default=200, 
                       help="Output resolution DPI")
    parser.add_argument("--format", type=str, choices=["png", "jpg", "pdf"], default="png", 
                       help="Output image format")
    parser.add_argument("--workers", type=int, default=4, 
                       help="Number of parallel worker processes")
    parser.add_argument("--no_probability_maps", action="store_true", 
                       help="Skip probability/confidence maps")
    
    # Mode selection
    parser.add_argument("--from_masks", action="store_true", 
                       help="Generate from saved masks instead of raw outputs")
    
    args = parser.parse_args()
    
    print("🎨 PIDNet Visualization Generator")
    print("=" * 50)
    
    # Validate arguments
    if args.from_masks and not args.image_dir:
        parser.error("--image_dir is required when using --from_masks")
    
    # Run visualization generation
    start_time = time.time()
    
    if args.from_masks:
        success = generate_visualizations_from_masks(
            inference_dir=args.dir,
            image_dir=args.image_dir,
            num_samples=args.samples,
            visualization_dpi=args.dpi,
            num_workers=args.workers,
            visualization_type=args.type,
            output_format=args.format
        )
    else:
        success = generate_visualizations_from_raw(
            inference_dir=args.dir,
            num_samples=args.samples,
            visualization_dpi=args.dpi,
            save_probability_maps=not args.no_probability_maps,
            num_workers=args.workers,
            visualization_type=args.type,
            output_format=args.format
        )
    
    total_time = time.time() - start_time
    
    if success:
        print(f"\n🎉 Visualization generation completed successfully!")
        print(f"⏱️  Total execution time: {total_time:.1f}s")
        print(f"🎯 Quality: Perect overlay alignment + Sharp mask boundaries")
        print(f"📁 Check your results directory for the  visualizations!")
    else:
        print(f"\n❌ Visualization generation failed!")
        print(f"💡 Check the error messages above and ensure your input files are valid")