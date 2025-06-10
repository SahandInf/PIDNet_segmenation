# PIDNet Segmentation Pipeline

High-performance semantic segmentation using PIDNet (Pixel-level Instance Discrimination Network) with PyTorch Lightning.

## Quick Start

### 1. Setup

#### Using pip (traditional method)
```bash
# Clone and install
git clone https://github.com/SahandInf/PIDNet_segmenation.git
cd PIDNet_segmenation
pip install -e .
```

#### Using uv (recommended - faster)
```bash
# Clone the repository
git clone https://github.com/SahandInf/PIDNet_segmenation.git
cd PIDNet_segmenation

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or sync from pyproject.toml directly
uv sync
```

#### Using Docker
```bash
docker-compose up -d
```

### 2. Quick Start with uv

```bash
# Activate environment
source .venv/bin/activate

# Run training
uv run python train.py

# Run inference
uv run python inference.py --image path/to/image.jpg

# Or without activation
uv run train.py
```

### 3. Prepare Dataset
```
datasets/your_dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### 4. Configure
Edit `config.yaml`:
```yaml
# Model Configuration
model:
  name: "pidnet-l"      # pidnet-s, pidnet-m, or pidnet-l (download from https://github.com/XuJiacong/PIDNet)
  in_channels: 3        # Number of input channels
  out_channels: 1       # Number of output channels (classes)
  weights_dir: "pretrained_models/imagenet"

# Training Configuration
training:
  batch_size: 8
  learning_rate: 0.0001
  epochs: 100
  early_stopping_patience: 20
  checkpoint_dir: 'checkpoints/'
  log_dir: 'logs/'

# Data Configuration
data:
  dataset_name: 'your_dataset'
  datasets_root: 'datasets/'
  img_size: [320, 128]  # [height, width]
  preserve_aspect_ratio: true
  num_workers: 8

# Inference Configuration
inference:
  checkpoint_path: 'path/to/best_model.ckpt'  # Will use latest if not specified
  output_dir: 'predictions/'
  
  # Performance Settings (optimized for RTX 4060 Ti)
  batch_size: 64        # Higher batch size for faster inference
  num_workers: 8
  mixed_precision: true # Enable AMP for speed boost
  
  # Output Settings
  save_visualizations: true
  save_masks: true
  save_report: true
  visualization_dpi: 200  # Lower DPI for faster saving
```

### 5. Train
```bash
python train.py --config config.yaml --name my_experiment

# Monitor progress
tensorboard --logdir logs/
```

### 6. Inference

**Single/Batch inference with visualizations:**
```bash
# Uses inference config from config.yaml
python inference.py --config config.yaml --dir path/to/images

# Override specific settings
python inference.py --config config.yaml --dir path/to/images --batch_size 32
```

**High-speed bulk inference (masks only):**
```bash
# Automatically uses optimized settings from inference config
python bulk_inference_only.py --config config.yaml --dir path/to/images --recursive
```

**Generate visualizations from bulk inference:**
```bash
# From raw outputs
python generate_visualizations.py --dir inference_results/run_dir

# From saved masks
python generate_visualizations.py --dir inference_results/run_dir --image_dir path/to/original/images --from_masks
```

## Key Features

- **High-quality output**: Proper padding handling preserves mask quality at original resolution
- **Flexible inference**: Single images, directories, or massive datasets
- **Performance optimized**: Configurable batch processing, multi-threading, mixed precision
- **Comprehensive reports**: HTML visualizations, metrics, and analysis
- **Docker support**: GPU-enabled container with TensorBoard

## Inference Configuration Details

The `inference` section in `config.yaml` controls behavior during inference:

- **checkpoint_path**: Path to trained model (auto-finds latest if not specified)
- **output_dir**: Where to save results
- **batch_size**: Higher values = faster processing (adjust based on GPU memory)
- **mixed_precision**: Enable for ~2x speedup on modern GPUs
- **save_visualizations**: Toggle visualization generation (disable for speed)
- **visualization_dpi**: Lower values = faster saving

## Common Tasks

**Process large dataset with custom settings:**
```bash
# Override config settings via command line
python bulk_inference_only.py --config config.yaml \
  --dir /data/images \
  --batch_size 128 \
  --workers 16
```

**Create mixed dataset:**
```bash
python create_mixed_dataset.py --datasets-root datasets --output-name combined
```

**Validate model:**
```bash
python model_validation.py config.yaml
```

## Output Structure

```
predictions/
├── masks/                    # High-quality segmentation masks
├── visualizations/           # Optional visualizations
├── inference_report.html     # Detailed report
└── config.json              # Used configuration
```