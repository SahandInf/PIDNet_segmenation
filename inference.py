import os
import argparse
import torch
import glob
import json
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import torch.nn.functional as F

from models import SegmentationModel
from utils import load_yaml_config, create_run_directory


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


def visualize_predictions(image, mask, prediction, threshold=0.5):
    """Visualize image, mask and prediction for inference."""
    # Ensure all tensors are on CPU
    if image.device.type != 'cpu':
        image = image.cpu()
    if mask.device.type != 'cpu':
        mask = mask.cpu()
    if prediction.device.type != 'cpu':
        prediction = prediction.cpu()
    
    # Denormalize image
    img = image.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # Process mask
    mask = mask.numpy().squeeze()
    
    # Process prediction
    if prediction.shape[0] > 1:  # Multi-class
        prediction = torch.softmax(prediction, dim=0)
        pred_probs = prediction.numpy()  # Save probabilities for visualization
        prediction = torch.argmax(prediction, dim=0).numpy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Plot original image
        axes[0].imshow(img)
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        # Plot ground truth mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Plot prediction
        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Plot prediction probability
        # Safely get the prediction class and ensure it's within bounds
        pred_classes = np.unique(prediction)
        if len(pred_classes) > 0:
            # Use the most common class for visualization
            from scipy import stats
            pred_class = stats.mode(prediction.flatten(), keepdims=False)[0]
            if pred_class < pred_probs.shape[0]:  # Ensure class index is within bounds
                prob_map = pred_probs[pred_class]
                im = axes[3].imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
                axes[3].set_title(f'Class {pred_class} Probability')
            else:
                # Fallback if class is out of bounds
                im = axes[3].imshow(np.zeros_like(prediction), cmap='viridis', vmin=0, vmax=1)
                axes[3].set_title('No Valid Probability Map')
        else:
            # Fallback if no classes are found
            im = axes[3].imshow(np.zeros_like(prediction), cmap='viridis', vmin=0, vmax=1)
            axes[3].set_title('No Classes Detected')
    else:  # Binary
        # Apply sigmoid to get probability values
        prediction = torch.sigmoid(prediction)
        
        # FIXED: Removed the manual inversion to match training behavior
        # prediction = 1.0 - prediction  <- This line was removed
        
        pred_probs = prediction.numpy().squeeze()  # Save probability for visualization
        prediction = (prediction > threshold).float().numpy().squeeze()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Plot original image
        axes[0].imshow(img)
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        # Plot ground truth mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Plot prediction
        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Plot probability map for binary segmentation
        im = axes[3].imshow(pred_probs, cmap='viridis', vmin=0, vmax=1)
        axes[3].set_title('Probability Map')
    
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def inference(config_path, image_path=None, image_dir=None, batch_size=1, output_dir=None, save_report=True):
    """Run inference using a trained PIDNet model with detailed reporting."""
    # Load configuration
    config = load_yaml_config(config_path)
    
    # If image_dir is not specified but we have a dataset name, use the test directory
    if image_dir is None and image_path is None and "data" in config and "dataset_name" in config["data"]:
        dataset_path = os.path.join(config["data"]["datasets_root"], config["data"]["dataset_name"])
        test_path = os.path.join(dataset_path, "test", "images")
        if os.path.exists(test_path):
            image_dir = test_path
            print(f"No image directory specified, using test set from dataset: {config['data']['dataset_name']}")
    
    # Create output directory
    if output_dir:
        inference_dir = output_dir
    else:
        inference_dir, run_name = create_run_directory(
            base_dir=config["inference"]["output_dir"],
            experiment_name="inference"
        )
    
    os.makedirs(inference_dir, exist_ok=True)
    predictions_dir = os.path.join(inference_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Save a copy of the config
    with open(os.path.join(inference_dir, "inference_config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Get preserve_aspect_ratio setting from config (default to True for consistency with training)
    preserve_aspect_ratio = config["data"].get("preserve_aspect_ratio", True)
    
    # Determine target image size
    if isinstance(config["data"]["img_size"], list) or isinstance(config["data"]["img_size"], tuple):
        # If img_size is provided as [height, width]
        img_height, img_width = config["data"]["img_size"]
        img_size = (img_width, img_height)  # PIL uses (width, height)
        print(f"Using rectangular image size: height={img_height}, width={img_width}")
    else:
        # If img_size is a single value
        img_height = img_width = config["data"]["img_size"]
        img_size = (img_width, img_height)  # PIL uses (width, height)
        print(f"Using square image size: {img_height}x{img_width}")
    
    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n=== INFERENCE SESSION ===")
    print(f"Using device: {device}")
    print(f"Model checkpoint: {config['inference']['checkpoint_path']}")
    print(f"Output directory: {inference_dir}")
    print(f"Preserve aspect ratio: {preserve_aspect_ratio}")
    
    # Print dataset information if available
    if "data" in config and "dataset_name" in config["data"]:
        print(f"Dataset: {config['data']['dataset_name']}")
    
    try:
        model = SegmentationModel.load_from_checkpoint(
            config["inference"]["checkpoint_path"],
            config=config
        )
        # Extract model info from checkpoint if available
        if hasattr(model, "hparams"):
            model_info = {
                "type": f"PIDNet-{model.hparams.config['model']['name'][-1].upper()}",
                "in_channels": model.hparams.config["model"]["in_channels"],
                "out_channels": model.hparams.config["model"]["out_channels"],
            }
        else:
            model_info = {
                "type": f"PIDNet-{config['model']['name'][-1].upper()}",
                "in_channels": config["model"]["in_channels"],
                "out_channels": config["model"]["out_channels"],
            }
        
        print(f"Model architecture: {model_info['type']} with {model_info['in_channels']} input channels, "
              f"{model_info['out_channels']} output channels")
        
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Prepare image paths
    if image_path:
        image_paths = [image_path]
    elif image_dir:
        image_paths = glob.glob(os.path.join(image_dir, "*.png")) + \
                      glob.glob(os.path.join(image_dir, "*.jpg")) + \
                      glob.glob(os.path.join(image_dir, "*.jpeg"))
    else:
        raise ValueError("Either --image or --dir must be provided.")
    
    print(f"Found {len(image_paths)} images for inference")
    
    # Initialize metrics tracking
    dice_scores = []
    inference_times = []
    results = []
    
    # Run inference on each image
    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            try:
                # Load and preprocess image
                print(f"Processing image {idx+1}/{len(image_paths)}: {os.path.basename(img_path)}")
                image = Image.open(img_path).convert('RGB')
                original_size = image.size
                print(f"  - Original image size: {original_size}")
                
                # Process image using the SAME approach as in training
                if preserve_aspect_ratio:
                    # Use unified aspect ratio preserving function
                    input_tensor = process_image_preserving_aspect_ratio(
                        image, 
                        img_size, 
                        is_mask=False, 
                        to_tensor=True, 
                        normalize=True
                    )
                else:
                    # Use standard resize (but still matching training pipeline order)
                    image_resized = image.resize(img_size, Image.BILINEAR)
                    input_tensor = transforms.ToTensor()(image_resized)
                    input_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input_tensor)
                
                # Add batch dimension and move to device
                input_tensor = input_tensor.unsqueeze(0).to(device)
                transformed_size = (input_tensor.shape[2], input_tensor.shape[3])  # Height, Width
                print(f"  - Transformed to: {transformed_size}")
                
                # Measure inference time
                start_time = time.time()
                
                # Forward pass
                output = model(input_tensor)
                
                # PIDNet with augment=True returns a list of outputs
                if isinstance(output, list):
                    # The second element is the main segmentation output
                    output = output[1]
                
                # Important: Resize output to match the expected size for masks
                output = F.interpolate(output, size=transformed_size, mode='bilinear', align_corners=True)
                
                # Measure end time
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Important: Move output tensor to CPU before further processing
                output = output.cpu().squeeze(0)  # Remove batch dimension and move to CPU
                
                # Try to find corresponding mask if it exists (for visualization and metrics)
                has_ground_truth = False
                
                # Check multiple possible mask paths
                potential_mask_paths = [
                    # Standard paths
                    img_path.replace('/images/', '/masks/').replace('.jpg', '_mask.png').replace('.png', '_mask.png'),
                    img_path.replace('/images/', '/masks/'),
                    
                    # For dataset structure: datasets/dataset_name/test/masks/
                    os.path.join(os.path.dirname(os.path.dirname(img_path)), 'masks', os.path.basename(img_path)),
                    os.path.join(os.path.dirname(os.path.dirname(img_path)), 'masks', 
                                os.path.splitext(os.path.basename(img_path))[0] + '_mask.png')
                ]
                
                for mask_path in potential_mask_paths:
                    if os.path.exists(mask_path):
                        has_ground_truth = True
                        mask = Image.open(mask_path).convert('L')
                        
                        # Process the mask using the unified function
                        if preserve_aspect_ratio:
                            mask_tensor = process_image_preserving_aspect_ratio(
                                mask, 
                                img_size, 
                                is_mask=True, 
                                to_tensor=True, 
                                normalize=False
                            )
                        else:
                            mask = mask.resize(img_size, Image.NEAREST)
                            mask_tensor = transforms.ToTensor()(mask)
                        
                        print(f"  - Found ground truth mask: {mask_path}")
                        break
                
                if not has_ground_truth:
                    # Create an empty mask if no ground truth exists
                    mask_tensor = torch.zeros((1, transformed_size[0], transformed_size[1]))
                    print(f"  - No ground truth mask found")
                
                # Calculate Dice score if ground truth is available
                if has_ground_truth:
                    # Move both tensors to CPU for metric calculation
                    if config["model"]["out_channels"] > 1:
                        # Multi-class
                        pred = torch.softmax(output, dim=0)
                        pred_class = torch.argmax(pred, dim=0)
                        pred_one_hot = torch.zeros_like(pred)
                        pred_one_hot.scatter_(0, pred_class.unsqueeze(0), 1)
                        
                        # Convert mask to one-hot if needed
                        if mask_tensor.shape[0] == 1:
                            mask_class = mask_tensor.long()
                            mask_one_hot = torch.zeros((config["model"]["out_channels"], mask_tensor.shape[1], mask_tensor.shape[2]))
                            mask_one_hot.scatter_(0, mask_class, 1)
                        else:
                            mask_one_hot = mask_tensor
                        
                        # Calculate Dice for each class
                        dice_score = 0
                        for c in range(config["model"]["out_channels"]):
                            intersection = (pred_one_hot[c] * mask_one_hot[c]).sum()
                            union = pred_one_hot[c].sum() + mask_one_hot[c].sum()
                            if union > 0:
                                dice_score += (2.0 * intersection) / (union + 1e-6)
                        
                        dice_score /= config["model"]["out_channels"]
                    else:
                        # Binary - No inversion here to be consistent
                        pred = torch.sigmoid(output)
                        pred_binary = (pred > 0.5).float()
                        
                        # Make sure tensors are the same size before calculating intersection
                        assert pred_binary.shape == mask_tensor.shape, f"Prediction shape {pred_binary.shape} doesn't match mask shape {mask_tensor.shape}"
                        
                        intersection = (pred_binary * mask_tensor).sum()
                        union = pred_binary.sum() + mask_tensor.sum()
                        dice_score = (2.0 * intersection) / (union + 1e-6)
                    
                    dice_scores.append(dice_score.item())
                else:
                    dice_score = None
                
                # Make sure tensor is on CPU before visualization
                input_tensor_cpu = input_tensor.squeeze(0).cpu()
                
                # Visualize and save results
                fig = visualize_predictions(input_tensor_cpu, mask_tensor, output)
                
                # Extract filename without path and extension
                filename = os.path.splitext(os.path.basename(img_path))[0]
                output_path = os.path.join(predictions_dir, f"{filename}_prediction.png")
                fig.savefig(output_path, dpi=200)
                plt.close(fig)
                
                # Save the raw prediction as well
                if config["model"]["out_channels"] > 1:
                    pred = torch.softmax(output, dim=0)
                    pred = torch.argmax(pred, dim=0).cpu().numpy()
                else:
                    # FIXED: No inversion here anymore
                    pred = torch.sigmoid(output)
                    pred = (pred > 0.5).float().cpu().numpy().squeeze()
                
                pred_img = Image.fromarray((pred * 255).astype('uint8'))
                
                # Resize back to original dimensions
                pred_img = pred_img.resize(original_size, Image.BILINEAR)  # or Image.LANCZOS for even better quality)
                pred_img.save(os.path.join(predictions_dir, f"{filename}_segmentation.png"))
                
                # Store result information
                result = {
                    "image_path": img_path,
                    "output_path": output_path,
                    "segmentation_path": os.path.join(predictions_dir, f"{filename}_segmentation.png"),
                    "inference_time": inference_time,
                    "dice_score": dice_score.item() if dice_score is not None else None,
                    "has_ground_truth": has_ground_truth,
                    "original_size": original_size
                }
                results.append(result)
                
                print(f"  - Inference time: {inference_time:.4f}s")
                if dice_score is not None:
                    print(f"  - Dice score: {dice_score.item():.4f}")
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                import traceback
                traceback.print_exc()  # Print the full stack trace for better debugging
    
    # Create inference report
    if save_report and results:  # Only create report if there are results
        create_inference_report(inference_dir, results, dice_scores, inference_times, config, model_info)
        
    # Return summary
    return {
        "results": results,
        "avg_dice": np.mean(dice_scores) if dice_scores else None,
        "avg_inference_time": np.mean(inference_times) if inference_times else 0,
        "output_dir": inference_dir
    }


def create_inference_report(output_dir, results, dice_scores, inference_times, config, model_info):
    """Create a detailed inference report with metrics and visualizations."""
    # Check if there are any results
    if not results:
        print("No successful inferences to report.")
        return None
        
    # Create a comprehensive HTML report
    report_path = os.path.join(output_dir, "inference_report.html")
    
    # Calculate summary statistics
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    std_inference_time = np.std(inference_times) if inference_times else 0
    
    if dice_scores:
        avg_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        min_dice = np.min(dice_scores)
        max_dice = np.max(dice_scores)
    else:
        avg_dice, std_dice, min_dice, max_dice = None, None, None, None
    
    # Generate HTML report
    with open(report_path, "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PIDNet Segmentation Inference Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
                .metric-card {{ background: #f8f9fa; border-radius: 8px; padding: 15px; min-width: 200px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
                .image-card {{ background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .image-card img {{ width: 100%; height: auto; }}
                .image-info {{ padding: 15px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>PIDNet Segmentation Inference Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Model Information</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Model Type</h3>
                        <div class="metric-value">{model_info['type']}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Input Channels</h3>
                        <div class="metric-value">{model_info['in_channels']}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Output Channels</h3>
                        <div class="metric-value">{model_info['out_channels']}</div>
                    </div>
                </div>
                
                <h2>Inference Summary</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Images Processed</h3>
                        <div class="metric-value">{len(results)}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Preserve Aspect Ratio</h3>
                        <div class="metric-value">{config['data'].get('preserve_aspect_ratio', True)}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Avg. Inference Time</h3>
                        <div class="metric-value">{avg_inference_time:.4f}s</div>
                        <p>±{std_inference_time:.4f}s</p>
                    </div>
        """)
        
        # Add Dice score metrics if available
        if dice_scores:
            f.write(f"""
                    <div class="metric-card">
                        <h3>Avg. Dice Score</h3>
                        <div class="metric-value">{avg_dice:.4f}</div>
                        <p>±{std_dice:.4f}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Min/Max Dice</h3>
                        <div class="metric-value">{min_dice:.4f} / {max_dice:.4f}</div>
                    </div>
            """)
            
        f.write(f"""
                </div>
                
                <h2>Detailed Results</h2>
                <table>
                    <tr>
                        <th>Image</th>
                        <th>Inference Time (s)</th>
                        <th>Dice Score</th>
                        <th>Actions</th>
                    </tr>
        """)
        
        # Add a row for each result
        for i, result in enumerate(results):
            filename = os.path.basename(result["image_path"])
            dice_value = f"{result['dice_score']:.4f}" if result['dice_score'] is not None else "N/A"
            
            f.write(f"""
                    <tr>
                        <td>{filename}</td>
                        <td>{result['inference_time']:.4f}</td>
                        <td>{dice_value}</td>
                        <td>
                            <a href="{os.path.relpath(result['output_path'], output_dir)}" target="_blank">View Result</a> | 
                            <a href="{os.path.relpath(result['segmentation_path'], output_dir)}" target="_blank">View Mask</a>
                        </td>
                    </tr>
            """)
        
        f.write(f"""
                </table>
                
                <h2>Result Visualizations</h2>
                <div class="image-grid">
        """)
        
        # Add thumbnails of results
        for i, result in enumerate(results):
            filename = os.path.basename(result["image_path"])
            dice_value = f"Dice: {result['dice_score']:.4f}" if result['dice_score'] is not None else "No Ground Truth"
            
            f.write(f"""
                    <div class="image-card">
                        <img src="{os.path.relpath(result['output_path'], output_dir)}" alt="Prediction for {filename}" />
                        <div class="image-info">
                            <h3>{filename}</h3>
                            <p>{dice_value}</p>
                            <p>Inference Time: {result['inference_time']:.4f}s</p>
                        </div>
                    </div>
            """)
        
        f.write(f"""
                </div>
            </div>
        </body>
        </html>
        """)
    
    # Also create a JSON report
    json_report = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model_info": model_info,
        "config": config,
        "metrics": {
            "images_processed": len(results),
            "avg_inference_time": float(avg_inference_time),
            "std_inference_time": float(std_inference_time),
        },
        "results": results
    }
    
    if dice_scores:
        json_report["metrics"]["avg_dice"] = float(avg_dice)
        json_report["metrics"]["std_dice"] = float(std_dice)
        json_report["metrics"]["min_dice"] = float(min_dice)
        json_report["metrics"]["max_dice"] = float(max_dice)
    
    with open(os.path.join(output_dir, "inference_report.json"), "w") as f:
        json.dump(json_report, f, indent=4)
    
    # Create a simple text summary
    with open(os.path.join(output_dir, "inference_summary.txt"), "w") as f:
        f.write("=== PIDNet Segmentation Inference Summary ===\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_info['type']} with {model_info['in_channels']} input channels, ")
        f.write(f"{model_info['out_channels']} output channels\n\n")
        
        f.write(f"Images processed: {len(results)}\n")
        f.write(f"Average inference time: {avg_inference_time:.4f}s ± {std_inference_time:.4f}s\n")
        
        if dice_scores:
            f.write(f"Average Dice score: {avg_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"Dice score range: {min_dice:.4f} - {max_dice:.4f}\n\n")
        
        f.write("=== End of Summary ===\n")
    
    print(f"\nInference report saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained PIDNet model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--image", type=str, help="Path to single image for inference")
    parser.add_argument("--dir", type=str, help="Path to directory containing images for inference")
    parser.add_argument("--output", type=str, help="Custom output directory for inference results")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--report", action="store_true", help="Generate a detailed HTML report")
    args = parser.parse_args()
    
    if not args.image and not args.dir:
        parser.error("Either --image or --dir must be provided.")
    
    # Measure total execution time
    start_time = time.time()
    
    # Run inference
    results = inference(
        config_path=args.config,
        image_path=args.image,
        image_dir=args.dir,
        batch_size=args.batch_size,
        output_dir=args.output,
        save_report=args.report
    )
    
    # Print execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")
    
    # Print summary
    if results:
        print("\n=== INFERENCE SUMMARY ===")
        print(f"Images processed: {len(results['results'])}")
        print(f"Average inference time: {results['avg_inference_time']:.4f}s")
        
        if results['avg_dice'] is not None:
            print(f"Average Dice score: {results['avg_dice']:.4f}")
        
        print(f"Results saved to: {results['output_dir']}")
        
        # Suggest viewing the report
        if args.report:
            print(f"\nTo view the detailed report, open:")
            print(f"{os.path.join(results['output_dir'], 'inference_report.html')}")