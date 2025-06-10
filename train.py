import os
import argparse
import time
import json
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from models import SegmentationModel
from data import SegmentationDataModule
from utils import load_yaml_config, save_config_to_run

import torch
torch.set_float32_matmul_precision('medium')


def train(config_path, experiment_name=None):
    """Train PIDNet segmentation model using PyTorch Lightning with enhanced logging and checkpointing."""
    # Load configuration
    config = load_yaml_config(config_path)
    
    # Create a unique run ID with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = f"pidnet_segmentation_{timestamp}"
    else:
        experiment_name = f"{experiment_name}_{timestamp}"
    
    # Create run directories
    run_dir = os.path.join(config["training"]["log_dir"], experiment_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Save configuration for reproducibility
    save_config_to_run(config, run_dir)
    
    print(f"====================== TRAINING SESSION ======================")
    print(f"Run name: {experiment_name}")
    print(f"Run directory: {run_dir}")
    print(f"Started at: {timestamp}")
    print(f"Model: PIDNet-{config['model']['name'][-1].upper()} (in_channels: {config['model']['in_channels']}, "
        f"out_channels: {config['model']['out_channels']})")
    print(f"Dataset: {config['data']['dataset_name']}")
    print(f"Training data: {os.path.join(config['data']['datasets_root'], config['data']['dataset_name'], 'train')}")
    print(f"Validation data: {os.path.join(config['data']['datasets_root'], config['data']['dataset_name'], 'val')}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Max epochs: {config['training']['epochs']}")
    print(f"============================================================")
    
    # Initialize data module
    data_module = SegmentationDataModule(config)
    
    # Initialize model
    model = SegmentationModel(config)
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="pidnet-{epoch:03d}-val_dice_{val_dice_epoch:.4f}",
        monitor="val_dice_epoch",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_dice_epoch",
        mode="max",
        patience=config["training"]["early_stopping_patience"],
        verbose=True,
        min_delta=0.001
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Progress bar
    progress_bar = TQDMProgressBar(refresh_rate=20)
    
    # Initialize logger
    logger = TensorBoardLogger(
        save_dir=tensorboard_dir,
        name=experiment_name,
        version=None,
        default_hp_metric=False
    )
    
    # Log hyperparameters
    logger.log_hyperparams({
        "model_type": f"PIDNet-{config['model']['name'][-1].upper()}",
        "in_channels": config["model"]["in_channels"],
        "out_channels": config["model"]["out_channels"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "img_size": config["data"]["img_size"]
    })
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor, progress_bar],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
        accelerator="auto",  # Use GPU if available, CPU otherwise
        devices="auto",      # Use all available devices
        enable_progress_bar=True,
        enable_model_summary=True,
        profiler="simple",  # Add performance profiling
        precision="32",     # Try explicitly setting precision
        gradient_clip_val=config["training"]["gradient_clip_val"]
    )
    
    # Initialize timers
    start_time = time.time()
    
    # Train model
    print(f"\nStarting training...")
    trainer.fit(model, data_module)
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining completed in {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")
    
    # Test model
    print(f"\nEvaluating model on test set...")
    test_start_time = time.time()
    test_results = trainer.test(model, data_module, ckpt_path="best")
    test_time = time.time() - test_start_time
    
    # Save test results to the run directory
    with open(os.path.join(run_dir, "test_results.json"), "w") as f:
        json.dump(test_results[0], f, indent=4)
    
    # Print summary
    print(f"\n================= TRAINING SUMMARY =================")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation Dice score: {checkpoint_callback.best_model_score:.4f}")
    print(f"Test Dice score: {test_results[0]['test_dice_epoch']:.4f}")
    print(f"Test loss: {test_results[0]['test_loss']:.4f}")
    print(f"Training took {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")
    print(f"Evaluation took {test_time:.2f}s")
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"Run directory: {run_dir}")
    print(f"Test report directory: {os.path.join(run_dir, 'test_report')}")
    print(f"=====================================================")
    
    # Create a symbolic link to the best model for easy access
    best_model_link = os.path.join(run_dir, "best_model.ckpt")
    if os.path.exists(best_model_link):
        os.remove(best_model_link)
    os.symlink(os.path.abspath(checkpoint_callback.best_model_path), best_model_link)
    
    # Update config file with the best model path for easy inference
    config["inference"]["checkpoint_path"] = os.path.abspath(best_model_link)
    save_config_to_run(config, run_dir, filename="config_with_best_model.yaml")
    
    return checkpoint_callback.best_model_path, run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PIDNet segmentation model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--name", type=str, default=None, help="Experiment name (timestamp will be added)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with limited data")
    args = parser.parse_args()
    
    if args.debug:
        print("Running in debug mode with 10 batches")
        # You could set up debug mode to use less data and fewer epochs
    
    best_model_path, run_dir = train(args.config, args.name)
    print(f"Training completed. Best model saved at: {best_model_path}")
    print(f"To view training metrics, run: tensorboard --logdir={run_dir}/tensorboard")