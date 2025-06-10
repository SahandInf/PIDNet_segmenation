#!/usr/bin/env python3
"""
Pure bulk inference script for PIDNet segmentation model.
Focuses solely on fast, efficient inference without visualization overhead.
FIXED: Restored high-quality mask generation using nearest neighbor interpolation.
"""

import os
import argparse
import torch
import glob
import json
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from datetime import datetime
import time
import torch.nn.functional as F
import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from models import SegmentationModel
from utils import load_yaml_config, create_run_directory


def process_image_preserving_aspect_ratio(image, target_size, is_mask=False, to_tensor=True, normalize=True):
    """Process an image while preserving aspect ratio"""
    width, height = image.size
    aspect_ratio = width / height
    
    target_w, target_h = target_size
    if aspect_ratio > target_w / target_h:
        new_width = target_w
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_h
        new_width = int(new_height * aspect_ratio)
    
    interpolation = Image.NEAREST if is_mask else Image.BILINEAR
    resized_img = image.resize((new_width, new_height), interpolation)
    
    pad_w = max(0, target_w - new_width)
    pad_h = max(0, target_h - new_height)
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    
    padded_img = Image.new(image.mode, target_size, 0)
    padded_img.paste(resized_img, (padding[0], padding[1]))
    
    # Create padding info dictionary
    padding_info = {
        'new_width': new_width,
        'new_height': new_height,
        'pad_left': padding[0],
        'pad_top': padding[1],
        'pad_right': padding[2],
        'pad_bottom': padding[3]
    }
    
    if to_tensor:
        if is_mask:
            return transforms.ToTensor()(padded_img), padding_info
        else:
            image_tensor = transforms.ToTensor()(padded_img)
            if normalize:
                image_tensor = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )(image_tensor)
            return image_tensor, padding_info
    else:
        return padded_img, padding_info


def load_images_batch(image_paths, batch_size=8, num_workers=4):
    """Load images in batches using parallel processing"""
    def load_single_image(path):
        try:
            return Image.open(path).convert('RGB'), path, None
        except Exception as e:
            return None, path, str(e)
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(load_single_image, batch_paths))
        
        successful = [(img, path) for img, path, err in results if img is not None]
        failed = [(path, err) for img, path, err in results if img is None]
        
        if failed:
            for path, err in failed:
                print(f"Failed to load {path}: {err}")
        
        if successful:
            yield successful


def bulk_inference_only(config_path, image_dir, output_dir=None, batch_size=None, 
                       save_report=None, max_images=None, recursive=None, 
                       file_extensions=None, save_masks=None, num_workers=None,
                       use_amp=None, save_raw_outputs=False):
    """
    Pure inference without any visualization - maximum speed and efficiency.
    """
    # Load configuration
    config = load_yaml_config(config_path)
    
    # Get parameters from config with command-line overrides
    inference_config = config.get("inference", {})
    
    if batch_size is None:
        batch_size = inference_config.get("batch_size", 8)
    if num_workers is None:
        num_workers = inference_config.get("num_workers", min(8, mp.cpu_count()))
    if use_amp is None:
        use_amp = inference_config.get("mixed_precision", True)
    if save_report is None:
        save_report = inference_config.get("save_report", True)
    if save_masks is None:
        save_masks = inference_config.get("save_masks", True)
    if recursive is None:
        recursive = inference_config.get("recursive_search", False)
    
    print(f"🚀 PURE INFERENCE MODE (HIGH QUALITY MASKS)")
    print(f"📋 Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Workers: {num_workers}")
    print(f"   Mixed precision: {use_amp}")
    print(f"   Save masks: {save_masks}")
    print(f"   Save raw outputs: {save_raw_outputs}")
    print(f"   Recursive search: {recursive}")
    print(f"   🎯 Mask quality: HIGH (using nearest neighbor interpolation)")
    
    # Setup file extensions
    if file_extensions is None:
        file_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    # Create output directory
    if output_dir:
        inference_dir = output_dir
    else:
        inference_dir, run_name = create_run_directory(
            base_dir=config.get("inference", {}).get("output_dir", "inference_results"),
            experiment_name="bulk_inference_only"
        )
    
    os.makedirs(inference_dir, exist_ok=True)
    predictions_dir = os.path.join(inference_dir, "masks")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Create raw outputs directory if needed
    raw_outputs_dir = None
    if save_raw_outputs:
        raw_outputs_dir = os.path.join(inference_dir, "raw_outputs")
        os.makedirs(raw_outputs_dir, exist_ok=True)
    
    # Save configuration copy
    with open(os.path.join(inference_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Determine target image size
    if isinstance(config["data"]["img_size"], list) or isinstance(config["data"]["img_size"], tuple):
        img_height, img_width = config["data"]["img_size"]
        img_size = (img_width, img_height)
    else:
        img_height = img_width = config["data"]["img_size"]
        img_size = (img_width, img_height)
    
    # Find all images
    print("\n📂 Scanning for images...")
    image_paths = []
    if recursive:
        for ext in file_extensions:
            image_paths.extend(glob.glob(os.path.join(image_dir, '**', f'*{ext}'), recursive=True))
    else:
        for ext in file_extensions:
            image_paths.extend(glob.glob(os.path.join(image_dir, f'*{ext}')))
    
    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]
    
    image_paths.sort()
    print(f"Found {len(image_paths)} images to process")
    
    # Setup device and optimizations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print("✅ CUDA optimizations enabled")
    
    # Load model
    try:
        model = SegmentationModel.load_from_checkpoint(
            config["inference"]["checkpoint_path"],
            config=config
        )
        model.to(device)
        model.eval()
        
        # Apply optimizations
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='reduce-overhead')
                print("✅ Model compiled with torch.compile")
            except Exception as e:
                print(f"⚠️  torch.compile not available: {e}")
        
        model_info = {
            "type": f"PIDNet-{config['model']['name'][-1].upper()}",
            "in_channels": config["model"]["in_channels"],
            "out_channels": config["model"]["out_channels"],
        }
        
        print(f"📊 Model: {model_info['type']} ({model_info['in_channels']}→{model_info['out_channels']} channels)")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Initialize tracking
    all_results = []
    total_inference_time = 0
    total_images_processed = 0
    
    # Process images in batches
    print(f"\n⚡ Processing {len(image_paths)} images...")
    overall_start = time.time()
    
    progress_bar = tqdm.tqdm(total=len(image_paths), desc="Inference", unit="img")
    
    try:
        with torch.no_grad():
            for batch_data in load_images_batch(image_paths, batch_size, num_workers):
                if not batch_data:
                    continue
                
                batch_images = [img for img, _ in batch_data]
                batch_paths = [path for _, path in batch_data]
                
                # Preprocess batch
                batch_tensors = []
                batch_padding_infos = []  # NEW: Store padding info for each image
                for image in batch_images:
                    if config["data"].get("preserve_aspect_ratio", True):
                        tensor, padding_info = process_image_preserving_aspect_ratio(  # CHANGED: Get padding info
                            image, img_size, is_mask=False, to_tensor=True, normalize=True
                        )
                        batch_padding_infos.append(padding_info)  # NEW: Store padding info
                    else:
                        resized = image.resize(img_size, Image.BILINEAR)
                        tensor = transforms.ToTensor()(resized)
                        tensor = transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        )(tensor)
                        batch_padding_infos.append(None)  # NEW: No padding info for non-aspect-preserving
                    batch_tensors.append(tensor)
                
                if batch_tensors:
                    batch_input = torch.stack(batch_tensors).to(device, non_blocking=True)
                    
                    # Inference
                    batch_start = time.time()
                    
                    if use_amp and device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            outputs = model(batch_input)
                    else:
                        outputs = model(batch_input)
                    
                    # Handle PIDNet output format
                    if isinstance(outputs, list):
                        outputs = outputs[1]  # Main segmentation output
                    
                    batch_inference_time = time.time() - batch_start
                    total_inference_time += batch_inference_time
                    
                    # Move to CPU
                    outputs_cpu = outputs.cpu()
                    
                    # Process each result
                    for i, (image, image_path) in enumerate(zip(batch_images, batch_paths)):
                        try:
                            output = outputs_cpu[i]
                            original_size = image.size
                            
                            # Create file paths
                            rel_path = os.path.relpath(image_path, image_dir)
                            rel_dir = os.path.dirname(rel_path)
                            filename_base = os.path.splitext(os.path.basename(image_path))[0]
                            
                            if rel_dir:
                                os.makedirs(os.path.join(predictions_dir, rel_dir), exist_ok=True)
                                mask_path = os.path.join(predictions_dir, rel_dir, f"{filename_base}_mask.png")
                            else:
                                mask_path = os.path.join(predictions_dir, f"{filename_base}_mask.png")
                            
                            # 🎯 FIXED: Save segmentation mask with HIGH QUALITY (nearest neighbor)
                            if save_masks:
                                # NEW: Remove padding if aspect ratio was preserved
                                if config["data"].get("preserve_aspect_ratio", True) and i < len(batch_padding_infos) and batch_padding_infos[i]:
                                    padding_info = batch_padding_infos[i]
                                    # Extract content area (remove padding)
                                    output_unpadded = output[
                                        :,
                                        padding_info['pad_top']:padding_info['pad_top']+padding_info['new_height'],
                                        padding_info['pad_left']:padding_info['pad_left']+padding_info['new_width']
                                    ]
                                    # Resize to original dimensions
                                    output_resized = F.interpolate(
                                        output_unpadded.unsqueeze(0),
                                        size=(original_size[1], original_size[0]),  # (height, width)
                                        mode='bilinear',
                                        align_corners=False
                                    ).squeeze(0)
                                else:
                                    # No padding, just resize to original
                                    output_resized = F.interpolate(
                                        output.unsqueeze(0),
                                        size=(original_size[1], original_size[0]),  # (height, width)
                                        mode='bilinear',
                                        align_corners=False
                                    ).squeeze(0)
                                
                                # Generate mask from resized output
                                if config["model"]["out_channels"] > 1:
                                    pred = torch.softmax(output_resized, dim=0)
                                    pred_mask = torch.argmax(pred, dim=0).cpu().numpy()
                                else:
                                    pred = torch.sigmoid(output_resized)
                                    pred_mask = (pred > 0.5).float().cpu().numpy().squeeze()
                                
                                # Convert to 8-bit image (already at original resolution)
                                mask_img = Image.fromarray((pred_mask * 255).astype('uint8'))
                                mask_img.save(mask_path)
                            
                            # Save raw outputs if requested
                            raw_output_path = None
                            if save_raw_outputs and raw_outputs_dir:
                                if rel_dir:
                                    raw_dir = os.path.join(raw_outputs_dir, rel_dir)
                                    os.makedirs(raw_dir, exist_ok=True)
                                    raw_output_path = os.path.join(raw_dir, f"{filename_base}_raw.pt")
                                else:
                                    raw_output_path = os.path.join(raw_outputs_dir, f"{filename_base}_raw.pt")
                                
                                # Calculate padding info for saving
                                padding_info = None
                                if config["data"].get("preserve_aspect_ratio", True):
                                    width, height = original_size
                                    aspect_ratio = width / height
                                    target_w, target_h = img_size
                                    
                                    if aspect_ratio > target_w / target_h:
                                        new_width = target_w
                                        new_height = int(new_width / aspect_ratio)
                                    else:
                                        new_height = target_h
                                        new_width = int(new_height * aspect_ratio)
                                    
                                    pad_w = max(0, target_w - new_width)
                                    pad_h = max(0, target_h - new_height)
                                    
                                    padding_info = {
                                        'new_width': new_width,
                                        'new_height': new_height,
                                        'pad_left': pad_w // 2,
                                        'pad_top': pad_h // 2,
                                        'pad_right': pad_w - pad_w // 2,
                                        'pad_bottom': pad_h - pad_h // 2
                                    }
                                
                                raw_data = {
                                    'output': output,
                                    'original_size': original_size,
                                    'image_path': image_path,
                                    'processed_size': img_size,
                                    'padding_info': padding_info  # NEW: Save padding info
                                }
                                torch.save(raw_data, raw_output_path)
                            
                            # Store result
                            result = {
                                "image_path": rel_path,
                                "mask_path": os.path.relpath(mask_path, inference_dir) if save_masks else None,
                                "raw_output_path": os.path.relpath(raw_output_path, inference_dir) if raw_output_path else None,
                                "inference_time": batch_inference_time / len(batch_images),
                                "original_size": original_size,
                                "processed_successfully": True
                            }
                            all_results.append(result)
                            total_images_processed += 1
                            
                        except Exception as e:
                            print(f"Error processing {image_path}: {e}")
                            all_results.append({
                                "image_path": os.path.relpath(image_path, image_dir),
                                "error": str(e),
                                "processed_successfully": False
                            })
                        
                        progress_bar.update(1)
                
                # Periodic cache clearing
                if device.type == 'cuda' and total_images_processed % 100 == 0:
                    torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    finally:
        progress_bar.close()
    
    # Calculate statistics
    total_time = time.time() - overall_start
    successful_results = [r for r in all_results if r.get("processed_successfully", False)]
    failed_results = [r for r in all_results if not r.get("processed_successfully", False)]
    
    avg_inference_time = total_inference_time / total_images_processed if total_images_processed > 0 else 0
    images_per_second = total_images_processed / total_time if total_time > 0 else 0
    
    # Print results
    print(f"\n{'='*60}")
    print(f"INFERENCE COMPLETED")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {len(successful_results)}")
    print(f"❌ Failed: {len(failed_results)}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"⚡ Speed: {images_per_second:.1f} images/second")
    print(f"🖥️  Avg inference time: {avg_inference_time*1000:.0f}ms per image")
    print(f"📁 Results saved to: {inference_dir}")
    print(f"🎯 Mask quality: HIGH (sharp edges preserved)")
    
    if save_raw_outputs:
        print(f"📦 Raw outputs saved for visualization")
        print(f"💡 Use generate_visualizations.py to create visualizations")
    
    # Create report
    if save_report:
        create_inference_report(inference_dir, all_results, {
            'total_time': total_time,
            'inference_time': total_inference_time,
            'avg_inference_time': avg_inference_time,
            'images_per_second': images_per_second,
            'successful_count': len(successful_results),
            'failed_count': len(failed_results),
            'batch_size': batch_size,
            'num_workers': num_workers,
            'use_amp': use_amp
        }, config, model_info)
        print(f"📊 Report: {os.path.join(inference_dir, 'inference_report.html')}")
    
    return {
        "results": all_results,
        "performance": {
            'total_time': total_time,
            'avg_inference_time': avg_inference_time,
            'images_per_second': images_per_second,
            'successful_count': len(successful_results),
            'failed_count': len(failed_results)
        },
        "output_dir": inference_dir,
        "raw_outputs_dir": raw_outputs_dir if save_raw_outputs else None
    }


def create_inference_report(output_dir, results, performance, config, model_info):
    """Create inference performance report"""
    report_path = os.path.join(output_dir, "inference_report.html")
    
    successful_results = [r for r in results if r.get("processed_successfully", False)]
    failed_results = [r for r in results if not r.get("processed_successfully", False)]
    
    with open(report_path, "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PIDNet Inference Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
                .header {{ background: linear-gradient(135deg, #3498db 0%, #2ecc71 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; border-radius: 4px; }}
                .metric-value {{ font-size: 28px; font-weight: bold; color: #3498db; }}
                .metric-label {{ color: #666; font-size: 14px; }}
                .success {{ color: #2ecc71; }}
                .error {{ color: #e74c3c; }}
                .config-section {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .quality-badge {{ background: #2ecc71; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚡ PIDNet Inference Report</h1>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Pure inference mode - Maximum speed processing</p>
                    <span class="quality-badge">🎯 HIGH QUALITY MASKS</span>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value success">{performance['successful_count']}</div>
                        <div class="metric-label">Images Processed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{performance['images_per_second']:.1f}</div>
                        <div class="metric-label">Images/Second</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{performance['avg_inference_time']*1000:.0f}ms</div>
                        <div class="metric-label">Avg Inference Time</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{performance['total_time']:.1f}s</div>
                        <div class="metric-label">Total Time</div>
                    </div>
                </div>
                
                <div class="config-section">
                    <h3>⚙️ Configuration</h3>
                    <ul>
                        <li><strong>Model:</strong> {model_info['type']} ({model_info['in_channels']}→{model_info['out_channels']} channels)</li>
                        <li><strong>Batch Size:</strong> {performance['batch_size']}</li>
                        <li><strong>Workers:</strong> {performance['num_workers']}</li>
                        <li><strong>Mixed Precision:</strong> {'✅ Enabled' if performance['use_amp'] else '❌ Disabled'}</li>
                        <li><strong>Image Size:</strong> {config['data']['img_size']}</li>
                        <li><strong>Mask Quality:</strong> <span class="quality-badge">HIGH (Nearest Neighbor)</span></li>
                    </ul>
                </div>
                
                <div class="config-section">
                    <h3>📊 Performance</h3>
                    <ul>
                        <li><strong>Pure Inference:</strong> {performance['inference_time']:.1f}s ({performance['inference_time']/performance['total_time']*100:.1f}%)</li>
                        <li><strong>I/O & Processing:</strong> {performance['total_time']-performance['inference_time']:.1f}s ({(performance['total_time']-performance['inference_time'])/performance['total_time']*100:.1f}%)</li>
                        <li><strong>Throughput:</strong> {performance['images_per_second']:.1f} images/second</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """)
    
    # JSON report
    json_report = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "mode": "inference_only_high_quality",
        "performance": performance,
        "model_info": model_info,
        "mask_quality": "high_nearest_neighbor",
        "results_summary": {
            "successful": performance['successful_count'],
            "failed": performance['failed_count'],
            "total": len(results)
        }
    }
    
    with open(os.path.join(output_dir, "inference_report.json"), "w") as f:
        json.dump(json_report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure PIDNet bulk inference - maximum speed, high quality masks")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config)")
    parser.add_argument("--max_images", type=int, help="Maximum images to process")
    parser.add_argument("--recursive", action="store_true", help="Search recursively")
    parser.add_argument("--no_report", action="store_true", help="Skip report generation")
    parser.add_argument("--no_masks", action="store_true", help="Skip mask saving")
    parser.add_argument("--workers", type=int, help="Number of worker threads")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--save_raw_outputs", action="store_true", help="Save raw outputs for visualization")
    
    args = parser.parse_args()
    
    print("⚡ Starting Pure Inference Mode (High Quality Masks)...")
    
    results = bulk_inference_only(
        config_path=args.config,
        image_dir=args.dir,
        output_dir=args.output,
        batch_size=args.batch_size,
        save_report=not args.no_report,
        max_images=args.max_images,
        recursive=args.recursive,
        save_masks=not args.no_masks,
        num_workers=args.workers,
        use_amp=not args.no_amp,
        save_raw_outputs=args.save_raw_outputs
    )
    
    if results:
        perf = results['performance']
        print(f"\n🎉 Inference completed!")
        print(f"📈 Speed: {perf['images_per_second']:.1f} images/second")
        print(f"📁 Results: {results['output_dir']}")
        print(f"🎯 Quality: HIGH (sharp mask edges preserved)")
        
        if args.save_raw_outputs:
            print(f"\n💡 To generate visualizations:")
            print(f"   python generate_visualizations.py --dir {results['output_dir']}")