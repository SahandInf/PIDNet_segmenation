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
model:
  name: "pidnet-l"      # pidnet-s, pidnet-m, or pidnet-l downloadeable here: https://github.com/XuJiacong/PIDNet
  out_channels: 1       # Number of classes

data:
  dataset_name: 'your_dataset'
  img_size: [160, 416]  # [height, width]
  preserve_aspect_ratio: true

training:
  batch_size: 8
  epochs: 100
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
python inference.py --config config.yaml --dir path/to/images
```

**High-speed bulk inference (masks only):**
```bash
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
- **Performance optimized**: Multi-threading, mixed precision, batch processing
- **Comprehensive reports**: HTML visualizations, metrics, and analysis
- **Docker support**: GPU-enabled container with TensorBoard

## Common Tasks

**Process large dataset:**
```bash
python bulk_inference_only.py --config config.yaml --dir /data/images --batch_size 16 --workers 8
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

