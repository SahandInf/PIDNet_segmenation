import os
import yaml
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shutil


def load_yaml_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config_to_run(config, run_dir, filename="config.yaml"):
    """Save configuration to run directory for reproducibility."""
    config_path = os.path.join(run_dir, filename)
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    # Also save as JSON for easier reading in other tools
    json_config_path = os.path.join(run_dir, os.path.splitext(filename)[0] + ".json")
    with open(json_config_path, 'w') as file:
        json.dump(config, file, indent=4)


def save_predictions(images, masks, predictions, save_dir, batch_idx, threshold=0.5):
    """Save prediction results as image files."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Move tensors to CPU and convert to numpy
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    
    # Process predictions based on number of channels
    if predictions.shape[1] > 1:  # Multi-class
        predictions = torch.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1).cpu().numpy()
    else:  # Binary
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > threshold).float().cpu().numpy()
    
    # Iterate through batch and save images
    for i in range(images.shape[0]):
        # Denormalize image
        img = images[i].transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Get mask and prediction
        mask = masks[i].squeeze()
        pred = predictions[i].squeeze()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        axes[0].imshow(img)
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        # Plot ground truth mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Plot prediction
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'batch{batch_idx}_sample{i}.png'), dpi=200)
        plt.close(fig)


def create_run_directory(base_dir, experiment_name=None):
    """Create a uniquely named run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name is None:
        run_name = f"run_{timestamp}"
    else:
        run_name = f"{experiment_name}_{timestamp}"
    
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir, run_name


def save_training_history(history, run_dir):
    """Save training history to CSV and generate plots."""
    # Save as CSV
    history_df = pd.DataFrame(history)
    history_path = os.path.join(run_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    
    # Create plots directory
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=history_df, x="epoch", y="train_loss", label="Train Loss")
    sns.lineplot(data=history_df, x="epoch", y="val_loss", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=200)
    plt.close()
    
    # Plot Dice coefficient
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=history_df, x="epoch", y="train_dice_epoch", label="Train Dice")
    sns.lineplot(data=history_df, x="epoch", y="val_dice_epoch", label="Validation Dice")
    plt.title("Training and Validation Dice Coefficient")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "dice_curve.png"), dpi=200)
    plt.close()
    
    # Plot learning rate
    if "learning_rate" in history_df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=history_df, x="epoch", y="learning_rate")
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "learning_rate.png"), dpi=200)
        plt.close()
    
    return plots_dir


def create_model_card(run_dir, config, model_info, test_results):
    """Create a detailed model card with metrics, configuration, and example images."""
    model_card_path = os.path.join(run_dir, "MODEL_CARD.md")
    
    with open(model_card_path, "w") as f:
        f.write(f"# U-Net Segmentation Model Card\n\n")
        f.write(f"## Model Information\n\n")
        f.write(f"- **Date Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Model Type:** U-Net for Image Segmentation\n")
        f.write(f"- **Version:** {model_info.get('version', '1.0.0')}\n")
        f.write(f"- **Framework:** PyTorch Lightning\n")
        f.write(f"- **License:** MIT\n\n")
        
        f.write(f"## Model Architecture\n\n")
        f.write(f"- **Input Channels:** {config['model']['in_channels']}\n")
        f.write(f"- **Output Channels:** {config['model']['out_channels']}\n")
        f.write(f"- **Initial Features:** {config['model']['init_features']}\n")
        f.write(f"- **Upsampling Type:** {'Bilinear' if config['model']['bilinear'] else 'Transposed Convolution'}\n\n")
        
        f.write(f"## Training Information\n\n")
        f.write(f"- **Training Data:** {config['data']['train_data_dir']}\n")
        f.write(f"- **Validation Data:** {config['data']['val_data_dir']}\n")
        f.write(f"- **Test Data:** {config['data']['test_data_dir']}\n")
        f.write(f"- **Image Size:** {config['data']['img_size']}x{config['data']['img_size']}\n")
        f.write(f"- **Batch Size:** {config['training']['batch_size']}\n")
        f.write(f"- **Learning Rate:** {config['training']['learning_rate']}\n")
        f.write(f"- **Weight Decay:** {config['training']['weight_decay']}\n")
        f.write(f"- **Training Duration:** {model_info.get('training_duration', 'Not recorded')}\n")
        f.write(f"- **Number of Epochs:** {model_info.get('epochs_trained', 'Not recorded')}\n\n")
        
        f.write(f"## Performance Metrics\n\n")
        f.write(f"- **Best Validation Dice Score:** {model_info.get('best_val_dice', 'Not recorded'):.4f}\n")
        f.write(f"- **Test Dice Score:** {test_results.get('test_dice_epoch', 'Not recorded'):.4f}\n")
        f.write(f"- **Test Loss:** {test_results.get('test_loss', 'Not recorded'):.4f}\n\n")
        
        f.write(f"## Model Usage\n\n")
        f.write(f"To use this model for inference:\n\n")
        f.write(f"```bash\n")
        f.write(f"python inference.py --config {os.path.join(run_dir, 'config.yaml')} --image path/to/image.png\n")
        f.write(f"```\n\n")
        
        f.write(f"## Limitations and Considerations\n\n")
        f.write(f"- This model was trained on specific data and may not generalize to all scenarios.\n")
        f.write(f"- For best results, ensure input images are similar to the training distribution.\n")
        f.write(f"- Consider fine-tuning on domain-specific data for specialized applications.\n")
    
    return model_card_path


def visualize_predictions(image, mask, prediction, threshold=0.5):
    """Visualize image, mask and prediction for inference."""
    # Denormalize image
    img = image.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # Process mask
    mask = mask.cpu().numpy().squeeze()
    
    # Process prediction
    if prediction.shape[0] > 1:  # Multi-class
        prediction = torch.softmax(prediction, dim=0)
        pred_probs = prediction.cpu().numpy()  # Save probabilities for visualization
        prediction = torch.argmax(prediction, dim=0).cpu().numpy()
    else:  # Binary
        prediction = torch.sigmoid(prediction)
        pred_probs = prediction.cpu().numpy().squeeze()  # Save probability for visualization
        prediction = (prediction > threshold).float().cpu().numpy().squeeze()
    
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
    if prediction.shape[0] > 1:  # Multi-class
        # Show the probability of the predicted class
        pred_class = prediction.argmax()
        prob_map = pred_probs[pred_class]
        im = axes[3].imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
        axes[3].set_title(f'Class {pred_class} Probability')
    else:  # Binary
        im = axes[3].imshow(pred_probs, cmap='viridis', vmin=0, vmax=1)
        axes[3].set_title('Probability Map')
    
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig