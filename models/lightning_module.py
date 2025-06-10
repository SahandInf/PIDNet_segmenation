import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision.utils import make_grid
import os
from datetime import datetime
import torch
import json
from PIL import Image
import scipy.stats as stats

from models.pidnet import get_pidnet_model


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks, which is less affected by class imbalance."""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        # Apply sigmoid for binary or softmax for multi-class
        if logits.shape[1] > 1:
            # Multi-class segmentation
            probs = F.softmax(logits, dim=1)
            
            # Convert targets to one-hot if needed
            if targets.shape[1] == 1 or len(targets.shape) == 3:
                targets_one_hot = F.one_hot(targets.squeeze(1).long(), num_classes=logits.shape[1])
                targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
            else:
                targets_one_hot = targets
            
            # Calculate Dice for each class and average
            dice = 0
            for i in range(logits.shape[1]):
                dice += self._binary_dice(probs[:, i], targets_one_hot[:, i])
            
            return 1 - dice / logits.shape[1]  # Return loss (1 - Dice coefficient)
        else:
            # Binary segmentation
            probs = torch.sigmoid(logits)
            return 1 - self._binary_dice(probs, targets)
    
    def _binary_dice(self, probs, targets):
        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        # Return Dice coefficient
        return (2.0 * intersection + self.smooth) / (union + self.smooth)


class ComboLoss(nn.Module):
    """
    Combination of BCE and Dice loss, which handles class imbalance effectively.
    alpha controls the balance between BCE and Dice loss.
    """
    def __init__(self, alpha=0.5, smooth=1.0, pos_weight=None):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.pos_weight = pos_weight  # Store pos_weight but don't create BCE yet
        self.smooth = smooth
        self.dice = DiceLoss(smooth=smooth)
        
    def forward(self, logits, targets):
        # Create BCE loss with pos_weight on the same device as the input tensors
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            bce = nn.BCEWithLogitsLoss()
            
        bce_loss = bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


class SegmentationModel(pl.LightningModule):
    """PyTorch Lightning module for PIDNet segmentation model with enhanced logging and visualization."""
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize PIDNet model
        self.model = get_pidnet_model(config)
        
        # Define loss function with class weighting for imbalanced datasets
        if config["model"]["out_channels"] > 1:
            # Multi-class segmentation
            self.criterion = nn.CrossEntropyLoss()
        else:
            # Binary segmentation with combo loss
            # A weight > 1 gives more importance to the positive class (foreground)
            pos_weight = torch.tensor([3.77])  # Adjust this value based on class imbalance
            
            # Use Combo Loss (BCE + Dice) - better for highly imbalanced segmentation
            self.criterion = ComboLoss(
                alpha=0.5,           # Balance between BCE and Dice (0.5 = equal weight)
                smooth=1.0,          # Smoothing factor for Dice loss
                pos_weight=pos_weight # Weight for positive class in BCE
            )
        
        # Metrics tracking
        self.train_dice = 0.0
        self.val_dice = 0.0
        self.test_dice = 0.0
        
        # For boundary detection head when using augment=True
        self.boundary_loss = nn.BCEWithLogitsLoss()
        
        # Store validation predictions for visualization
        self.val_step_outputs = []
        self.val_step_targets = []
        self.val_step_inputs = []
        
        # Store test outputs for visualization and metrics
        self.test_step_outputs = []
        self.test_step_targets = []
        self.test_step_inputs = []
        
        # Initialize epoch counters and history
        self.current_epoch_train_dice_values = []
        self.current_epoch_val_dice_values = []
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_dice_epoch': [],
            'val_dice_epoch': [],
            'learning_rate': []
        }

    def forward(self, x):
        return self.model(x)

    def _compute_dice_coefficient(self, pred, target, smooth=1e-6):
        """Compute Dice coefficient for segmentation evaluation."""
        
        if self.config["model"]["out_channels"] > 1:
            # Multi-class: get class predictions
            pred = F.softmax(pred, dim=1).argmax(dim=1)
            # Convert target to class indices if it's one-hot encoded
            if target.shape[1] > 1:
                target = target.argmax(dim=1)
            # Convert to one-hot for dice calculation
            pred_one_hot = F.one_hot(pred, num_classes=self.config["model"]["out_channels"]).permute(0, 3, 1, 2).float()
            target_one_hot = F.one_hot(target, num_classes=self.config["model"]["out_channels"]).permute(0, 3, 1, 2).float()
            
            # Calculate dice for each class and average
            dice = 0
            for i in range(self.config["model"]["out_channels"]):
                intersection = (pred_one_hot[:, i] * target_one_hot[:, i]).sum()
                union = pred_one_hot[:, i].sum() + target_one_hot[:, i].sum()
                dice += (2.0 * intersection + smooth) / (union + smooth)
            
            return dice / self.config["model"]["out_channels"]
        else:
            # Binary: apply sigmoid for probability
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            
            # Flatten tensors for dice calculation
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
            
            # Guard against division by zero if both prediction and target are empty
            if union.item() < smooth:
                return torch.tensor(1.0, device=pred.device)
                
            return (2.0 * intersection + smooth) / (union + smooth)

    def _shared_step(self, batch, batch_idx, stage):
        """Shared step for training, validation, and testing."""
        images, masks = batch
        
        # Forward pass
        outputs = self.forward(images)
        
        # PIDNet returns multiple outputs when augment=True
        if isinstance(outputs, list):
            # Main segmentation output is the second element
            main_output = outputs[1]
            aux_seg_output = outputs[0]  # Auxiliary segmentation output
            boundary_output = outputs[2]  # Boundary detection output
            
            # Add interpolation to resize outputs to match mask size
            main_output = F.interpolate(main_output, size=masks.shape[2:], mode='bilinear', align_corners=True)
            aux_seg_output = F.interpolate(aux_seg_output, size=masks.shape[2:], mode='bilinear', align_corners=True)
            boundary_output = F.interpolate(boundary_output, size=masks.shape[2:], mode='bilinear', align_corners=True)
            
            # Calculate main segmentation loss
            if self.config["model"]["out_channels"] == 1:
                main_loss = self.criterion(main_output, masks)
                aux_loss = self.criterion(aux_seg_output, masks)
            else:
                # For multi-class, we need to manipulate the target shape
                main_loss = self.criterion(main_output, masks.squeeze(1).long())
                aux_loss = self.criterion(aux_seg_output, masks.squeeze(1).long())
            
            # Generate boundary ground truth (simplified approach)
            # In practice, you'd use a more sophisticated method to generate boundary ground truth
            if masks.shape[1] == 1:  # Binary segmentation
                # Simple edge detection using gradient
                boundary_target = self._generate_boundary_targets(masks)
                boundary_loss = self.boundary_loss(boundary_output, boundary_target)
            else:
                # For multi-class, we can still use a binary boundary
                boundary_target = self._generate_boundary_targets(masks.argmax(dim=1, keepdim=True))
                boundary_loss = self.boundary_loss(boundary_output, boundary_target)
            
            # Total loss is weighted sum of the three losses
            # Weights can be adjusted as hyperparameters
            loss = 0.8 * main_loss + 0.1 * aux_loss + 0.1 * boundary_loss
            
            # Calculate Dice coefficient for main output
            dice = self._compute_dice_coefficient(main_output, masks)
        else:
            # If using augment=False during inference
            main_output = outputs
            
            # Add interpolation for single output case too
            main_output = F.interpolate(main_output, size=masks.shape[2:], mode='bilinear', align_corners=True)
            
            # Calculate loss
            if self.config["model"]["out_channels"] == 1:
                loss = self.criterion(main_output, masks)
            else:
                # For multi-class, we need to manipulate the target shape
                loss = self.criterion(main_output, masks.squeeze(1).long())
            
            # Calculate Dice coefficient
            dice = self._compute_dice_coefficient(main_output, masks)
        
        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_dice", dice, prog_bar=True)
        
        return {"loss": loss, "dice": dice, "logits": main_output}
    
    def _generate_boundary_targets(self, masks, kernel_size=3):
        """
        Generate boundary ground truth for boundary detection head.
        Uses a simple gradient-based method.
        """
        # Ensure masks are in the right format
        if masks.shape[1] != 1:
            masks = masks.argmax(dim=1, keepdim=True)
        
        # Padding to keep the size
        p = kernel_size // 2
        masks_padded = F.pad(masks, (p, p, p, p), mode='reflect')
        
        # Calculate gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=masks.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=masks.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(masks_padded.float(), sobel_x, padding=0)
        grad_y = F.conv2d(masks_padded.float(), sobel_y, padding=0)
        
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize and threshold
        boundary = (grad_mag > 0.05).float()
        
        return boundary

    def training_step(self, batch, batch_idx):
        result = self._shared_step(batch, batch_idx, "train")
        self.current_epoch_train_dice_values.append(result["dice"].item())
        return result

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        result = self._shared_step(batch, batch_idx, "val")
        self.current_epoch_val_dice_values.append(result["dice"].item())
        
        # Store examples for visualization (save a few examples)
        if batch_idx < 4:  # Store only first 4 batches to avoid memory issues
            # Get predictions
            logits = result["logits"]
            
            # Store for visualization
            self.val_step_outputs.append(logits.detach().cpu())
            self.val_step_targets.append(masks.detach().cpu())
            self.val_step_inputs.append(images.detach().cpu())
        
        return result

    def test_step(self, batch, batch_idx):
        images, masks = batch
        result = self._shared_step(batch, batch_idx, "test")
        
        # Store results for end of epoch processing
        self.test_step_outputs.append(result["logits"].detach().cpu())
        self.test_step_targets.append(masks.detach().cpu())
        self.test_step_inputs.append(images.detach().cpu())
        
        # Log batch-level metrics in detail during testing
        self.log(f"test_batch_{batch_idx}_dice", result["dice"], on_step=True, on_epoch=False)
        self.log(f"test_batch_{batch_idx}_loss", result["loss"], on_step=True, on_epoch=False)
        return result
    
    def on_train_epoch_start(self):
        # Reset epoch counters
        self.current_epoch_train_dice_values = []
        print(f"\nEpoch {self.current_epoch+1}/{self.trainer.max_epochs} starting...")
    
    def on_validation_epoch_start(self):
        # Reset epoch counters and storage
        self.current_epoch_val_dice_values = []
        self.val_step_outputs = []
        self.val_step_targets = []
        self.val_step_inputs = []
    
    def on_test_epoch_start(self):
        # Reset storage for test outputs
        self.test_step_outputs = []
        self.test_step_targets = []
        self.test_step_inputs = []
    
    def on_train_epoch_end(self):
        # Calculate average metrics for the epoch
        avg_dice = np.mean(self.current_epoch_train_dice_values)
        self.train_dice = avg_dice
        
        # Log to progress bar and tensorboard
        self.log("train_dice_epoch", avg_dice, prog_bar=True)
        
        # Store in history
        self.training_history['train_dice_epoch'].append(avg_dice)
        
        # Get current learning rate from optimizer
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.training_history['learning_rate'].append(current_lr)
        
        # Report progress
        print(f"Epoch {self.current_epoch+1} train dice: {avg_dice:.4f}")

    def on_validation_epoch_end(self):
        # Calculate average metrics for the epoch
        avg_dice = np.mean(self.current_epoch_val_dice_values)
        self.val_dice = avg_dice
        
        # Log to progress bar and tensorboard
        self.log("val_dice_epoch", avg_dice, prog_bar=True)
        
        # Store in history
        if len(self.training_history['epoch']) <= self.current_epoch:
            self.training_history['epoch'].append(self.current_epoch+1)
            
        # Store loss values (get from trainer)
        if hasattr(self.trainer, 'callback_metrics'):
            train_loss = self.trainer.callback_metrics.get('train_loss')
            val_loss = self.trainer.callback_metrics.get('val_loss')
            
            if train_loss is not None:
                self.training_history['train_loss'].append(train_loss.item())
            
            if val_loss is not None:
                self.training_history['val_loss'].append(val_loss.item())
        
        # Visualize predictions if available
        if len(self.val_step_outputs) > 0:
            self._log_prediction_images()
        
        # Report progress
        print(f"Epoch {self.current_epoch+1} validation dice: {avg_dice:.4f}")

    def on_test_epoch_end(self):
        # Process test results
        all_outputs = torch.cat(self.test_step_outputs, dim=0)
        all_targets = torch.cat(self.test_step_targets, dim=0)
        all_inputs = torch.cat(self.test_step_inputs, dim=0)
        
        # Move tensors to the same device (CPU is safer for final calculations)
        device = torch.device('cpu')
        all_outputs = all_outputs.to(device)
        all_targets = all_targets.to(device)
        all_inputs = all_inputs.to(device)
        
        # Calculate mean Dice score
        dice_scores = []
        for i in range(len(all_outputs)):
            dice = self._compute_dice_coefficient(all_outputs[i:i+1], all_targets[i:i+1])
            dice_scores.append(dice.item())
        
        avg_dice = np.mean(dice_scores)
        self.test_dice = avg_dice
        
        # Log overall test metrics
        self.log("test_dice_epoch", avg_dice, prog_bar=True)
        
        # Calculate and log loss
        if self.config["model"]["out_channels"] == 1:
            test_loss = self.criterion(all_outputs, all_targets)
        else:
            test_loss = self.criterion(all_outputs, all_targets.squeeze(1).long())
        
        self.log("test_loss", test_loss, prog_bar=True)
        
        # Create test visualizations (similar to validation)
        if len(self.test_step_inputs) > 0:
            self._log_test_prediction_images()
        
        # Generate a comprehensive test report similar to inference.py
        self._generate_test_report(all_inputs, all_targets, all_outputs, dice_scores)
        
        # Display a detailed test report
        print("\n" + "="*50)
        print("TEST RESULTS:")
        print(f"- Test Dice Score: {avg_dice:.4f}")
        print(f"- Test Loss: {test_loss:.4f}")
        print("="*50 + "\n")
        
    def _log_prediction_images(self):
        """Log prediction visualizations to TensorBoard."""
        # Combine inputs from multiple batches
        all_inputs = torch.cat(self.val_step_inputs, dim=0)
        all_targets = torch.cat(self.val_step_targets, dim=0)
        all_outputs = torch.cat(self.val_step_outputs, dim=0)
        
        # Select a subset of images to visualize (max 8)
        num_images = min(8, all_inputs.size(0))
        
        # Create a figure for each sample
        for idx in range(num_images):
            fig = plt.figure(figsize=(15, 5))
            
            # Original image
            img = all_inputs[idx].permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # Ground truth mask
            mask = all_targets[idx].squeeze().numpy()
            
            # Prediction
            logits = all_outputs[idx]
            if self.config["model"]["out_channels"] > 1:  # Multi-class
                pred = torch.softmax(logits, dim=0)
                pred = torch.argmax(pred, dim=0).numpy()
            else:  # Binary
                pred = torch.sigmoid(logits)
                pred = (pred > 0.5).float().squeeze().numpy()
            
            # Plot
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title('Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(pred, cmap='gray')
            plt.title('Prediction')
            plt.axis('off')
            
            # Log to TensorBoard
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_figure(f'predictions/sample_{idx}', fig, self.current_epoch)
            plt.close(fig)

    def _log_test_prediction_images(self):
        """Log test prediction visualizations to TensorBoard."""
        # Similar to validation visualization but for test set
        all_inputs = torch.cat(self.test_step_inputs, dim=0)
        all_targets = torch.cat(self.test_step_targets, dim=0)
        all_outputs = torch.cat(self.test_step_outputs, dim=0)
        
        num_images = min(8, all_inputs.size(0))
        
        for idx in range(num_images):
            fig = plt.figure(figsize=(15, 5))
            
            # Original image
            img = all_inputs[idx].permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # Ground truth mask
            mask = all_targets[idx].squeeze().numpy()
            
            # Prediction
            logits = all_outputs[idx]
            if self.config["model"]["out_channels"] > 1:  # Multi-class
                pred = torch.softmax(logits, dim=0)
                pred = torch.argmax(pred, dim=0).numpy()
            else:  # Binary
                pred = torch.sigmoid(logits)
                pred = (pred > 0.5).float().squeeze().numpy()
            
            # Plot
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title('Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(pred, cmap='gray')
            plt.title('Prediction')
            plt.axis('off')
            
            # Log to TensorBoard
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_figure(f'test_predictions/sample_{idx}', fig, self.current_epoch)
            plt.close(fig)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Get configuration parameters with defaults and convert to proper types
        lr = float(self.config["training"]["learning_rate"])
        weight_decay = float(self.config["training"]["weight_decay"])
        
        # Use Adam instead of SGD for more stable convergence
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Use a more sophisticated learning rate scheduler
        scheduler_type = self.config.get("training", {}).get("scheduler", "plateau")
        
        if scheduler_type == "cosine":
            # Cosine Annealing LR scheduler
            t_max = self.config["training"]["epochs"]
            scheduler = {
                "scheduler": CosineAnnealingLR(
                    optimizer, 
                    T_max=t_max, 
                    eta_min=lr * 0.01
                ),
                "interval": "epoch",
                "frequency": 1,
                "name": "cosine_lr"
            }
        else:
            # Default: ReduceLROnPlateau
            scheduler = {
                "scheduler": ReduceLROnPlateau(
                    optimizer, 
                    mode="max", 
                    factor=0.5, 
                    patience=5, 
                    verbose=True
                ),
                "monitor": "val_dice_epoch",
                "interval": "epoch",
                "frequency": 1,
                "name": "plateau_lr"
            }
        
        return [optimizer], [scheduler]
        
    def on_save_checkpoint(self, checkpoint):
        """Add custom data to the checkpoint."""
        # Save training history
        checkpoint["training_history"] = self.training_history
        
        # Save model configuration
        checkpoint["model_config"] = self.config
        
        # Save architecture summary
        model_summary = {
            "name": self.config["model"]["name"],
            "in_channels": self.config["model"]["in_channels"],
            "out_channels": self.config["model"]["out_channels"],
        }
        checkpoint["model_summary"] = model_summary
        
        # Auto-generate a test report if this is the best model
        if hasattr(self.trainer, 'checkpoint_callback') and self.trainer.checkpoint_callback is not None:
            if hasattr(self.trainer.checkpoint_callback, 'best_model_score'):
                current_score = self.trainer.callback_metrics.get('val_dice_epoch', None)
                if current_score is not None and self.trainer.checkpoint_callback.best_model_score is not None:
                    # Check if this is the best model so far
                    if current_score >= self.trainer.checkpoint_callback.best_model_score:
                        print("\nThis is the best model so far. Running test inference to generate a comprehensive report...")
                        # The model will be tested at the end of training via the test() method in train.py
                        checkpoint["is_best_model"] = True
                    else:
                        checkpoint["is_best_model"] = False
        
    def on_load_checkpoint(self, checkpoint):
        """Load custom data from checkpoint."""
        if "training_history" in checkpoint:
            self.training_history = checkpoint["training_history"]

    def _generate_test_report(self, all_inputs, all_targets, all_outputs, dice_scores):
        """Generate a comprehensive test report similar to inference.py."""
        # Create a test report directory in the same location as the checkpoint
        if hasattr(self.trainer, 'log_dir'):
            base_dir = self.trainer.log_dir
        elif hasattr(self.trainer.logger, 'log_dir'):
            base_dir = self.trainer.logger.log_dir
        else:
            base_dir = os.path.join('test_reports', f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        test_report_dir = os.path.join(base_dir, 'test_report')
        predictions_dir = os.path.join(test_report_dir, 'predictions')
        
        os.makedirs(test_report_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(test_report_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
            
        # Model info
        model_info = {
            "type": f"PIDNet-{self.config['model']['name'][-1].upper()}",
            "in_channels": self.config["model"]["in_channels"],
            "out_channels": self.config["model"]["out_channels"],
        }
        
        inference_times = [0.01] * len(all_inputs)  # Placeholder
        
        # Process each sample
        results = []
        num_samples = min(len(all_inputs), 20)  # Limit to 20 samples to avoid excessive processing
        
        print(f"\nGenerating detailed test report for {num_samples} samples...")
        
        for idx in range(num_samples):
            # Get image, mask, and prediction
            input_tensor = all_inputs[idx]
            mask_tensor = all_targets[idx]
            output = all_outputs[idx]
            
            # Visualize prediction
            fig = self._visualize_prediction(input_tensor, mask_tensor, output)
            
            # Save visualization
            output_path = os.path.join(predictions_dir, f'sample_{idx+1}_prediction.png')
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
            
            # Save segmentation mask
            if self.config["model"]["out_channels"] > 1:
                pred = torch.softmax(output, dim=0)
                pred = torch.argmax(pred, dim=0).cpu().numpy()
            else:
                pred = torch.sigmoid(output)
                pred = (pred > 0.5).float().cpu().numpy().squeeze()
            
            # Convert to image
            pred_img = Image.fromarray((pred * 255).astype('uint8'))
            segmentation_path = os.path.join(predictions_dir, f'sample_{idx+1}_segmentation.png')
            pred_img.save(segmentation_path)
            
            # Store result
            result = {
                "sample_index": idx,
                "output_path": output_path,
                "segmentation_path": segmentation_path,
                "dice_score": dice_scores[idx] if idx < len(dice_scores) else None,
            }
            results.append(result)
            
        # Calculate statistics
        avg_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        min_dice = np.min(dice_scores) if dice_scores else 0
        max_dice = np.max(dice_scores) if dice_scores else 0
        
        # Create HTML report
        self._create_html_report(
            test_report_dir, 
            results, 
            {
                "avg_dice": avg_dice, 
                "std_dice": std_dice, 
                "min_dice": min_dice, 
                "max_dice": max_dice,
                "avg_inference_time": avg_inference_time
            }, 
            model_info
        )
        
        # Create JSON report
        json_report = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "model_info": model_info,
            "metrics": {
                "images_processed": len(results),
                "avg_dice": float(avg_dice),
                "std_dice": float(std_dice),
                "min_dice": float(min_dice),
                "max_dice": float(max_dice),
            },
            "results": results
        }
        
        with open(os.path.join(test_report_dir, "test_report.json"), "w") as f:
            json.dump(json_report, f, indent=4)
        
        # Create text summary
        with open(os.path.join(test_report_dir, "test_summary.txt"), "w") as f:
            f.write("=== PIDNet Segmentation Test Summary ===\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {model_info['type']} with {model_info['in_channels']} input channels, ")
            f.write(f"{model_info['out_channels']} output channels\n\n")
            
            f.write(f"Test samples: {len(dice_scores)}\n")
            f.write(f"Average Dice score: {avg_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"Dice score range: {min_dice:.4f} - {max_dice:.4f}\n\n")
            
            f.write("=== End of Summary ===\n")
        
        print(f"Test report saved to: {test_report_dir}")
        
        return test_report_dir

    def _visualize_prediction(self, image, mask, prediction, threshold=0.5):
        """Visualize image, mask and prediction."""
        # Convert to numpy for visualization
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
            pred = torch.softmax(prediction, dim=0)
            pred_probs = pred.cpu().numpy()  # Save probabilities for visualization
            pred = torch.argmax(pred, dim=0).cpu().numpy()
            
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
            axes[2].imshow(pred, cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            # Plot prediction probability
            # Safely get the prediction class
            pred_classes = np.unique(pred)
            if len(pred_classes) > 0:
                # Use the most common class for visualization
                pred_class = stats.mode(pred.flatten(), keepdims=False)[0]
                if pred_class < pred_probs.shape[0]:  # Ensure class index is within bounds
                    prob_map = pred_probs[pred_class]
                    im = axes[3].imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
                    axes[3].set_title(f'Class {pred_class} Probability')
                else:
                    # Fallback if class is out of bounds
                    im = axes[3].imshow(np.zeros_like(pred), cmap='viridis', vmin=0, vmax=1)
                    axes[3].set_title('No Valid Probability Map')
            else:
                # Fallback if no classes are found
                im = axes[3].imshow(np.zeros_like(pred), cmap='viridis', vmin=0, vmax=1)
                axes[3].set_title('No Classes Detected')
                
        else:  # Binary
            # Apply sigmoid to get probability values
            pred = torch.sigmoid(prediction)
            pred_probs = pred.cpu().numpy().squeeze()  # Save probability for visualization
            pred = (pred > threshold).float().cpu().numpy().squeeze()
            
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
            axes[2].imshow(pred, cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            # Plot probability map for binary segmentation
            im = axes[3].imshow(pred_probs, cmap='viridis', vmin=0, vmax=1)
            axes[3].set_title('Probability Map')
        
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig

    def _create_html_report(self, output_dir, results, metrics, model_info):
        """Create a detailed HTML report with metrics and visualizations."""
        # Create a comprehensive HTML report
        report_path = os.path.join(output_dir, "test_report.html")
        
        # Extract metrics
        avg_dice = metrics["avg_dice"]
        std_dice = metrics["std_dice"]
        min_dice = metrics["min_dice"]
        max_dice = metrics["max_dice"]
        avg_inference_time = metrics["avg_inference_time"]
        
        # Generate HTML report
        with open(report_path, "w") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>PIDNet Segmentation Test Report</title>
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
                    <h1>PIDNet Segmentation Test Report</h1>
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
                    
                    <h2>Test Summary</h2>
                    <div class="metrics">
                        <div class="metric-card">
                            <h3>Images Processed</h3>
                            <div class="metric-value">{len(results)}</div>
                        </div>
                        <div class="metric-card">
                            <h3>Avg. Inference Time</h3>
                            <div class="metric-value">{avg_inference_time:.4f}s</div>
                        </div>
                        <div class="metric-card">
                            <h3>Avg. Dice Score</h3>
                            <div class="metric-value">{avg_dice:.4f}</div>
                            <p>±{std_dice:.4f}</p>
                        </div>
                        <div class="metric-card">
                            <h3>Min/Max Dice</h3>
                            <div class="metric-value">{min_dice:.4f} / {max_dice:.4f}</div>
                        </div>
                    </div>
                    
                    <h2>Detailed Results</h2>
                    <table>
                        <tr>
                            <th>Sample</th>
                            <th>Dice Score</th>
                            <th>Actions</th>
                        </tr>
            """)
            
            # Add a row for each result
            for i, result in enumerate(results):
                dice_value = f"{result['dice_score']:.4f}" if result['dice_score'] is not None else "N/A"
                
                f.write(f"""
                        <tr>
                            <td>Sample {i+1}</td>
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
                dice_value = f"Dice: {result['dice_score']:.4f}" if result['dice_score'] is not None else "No Score"
                
                f.write(f"""
                        <div class="image-card">
                            <img src="{os.path.relpath(result['output_path'], output_dir)}" alt="Prediction {i+1}" />
                            <div class="image-info">
                                <h3>Sample {i+1}</h3>
                                <p>{dice_value}</p>
                            </div>
                        </div>
                """)
            
            f.write(f"""
                    </div>
                </div>
            </body>
            </html>
            """)
            
        return report_path