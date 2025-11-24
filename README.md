# 96.51-Fashion-MNIST

**Class project for EECS 230P**

A comprehensive deep learning project for classifying Fashion-MNIST images using multiple state-of-the-art architectures. This project implements and compares 5 different models: BetterCNN, ResNet18, EfficientNet-B0, DenseNet-121, and Vision Transformer (ViT-Tiny), achieving **96.51% accuracy** with ensemble methods.

## 📋 Table of Contents

- [Features](#features)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Multiple Model Architectures**: Implements 5 different deep learning models for comparison
- **Advanced Training Techniques**:
  - Mixed Precision Training (FP16) for faster training
  - Data Augmentation (MixUp, CutMix, Random Erasing)
  - Learning Rate Scheduling (Warmup + Cosine Annealing)
  - Early Stopping with patience mechanism
  - Gradient Clipping for training stability
- **Comprehensive Evaluation**:
  - Test-Time Augmentation (TTA) support
  - Model ensemble methods (Simple Average, Weighted Average, Geometric Mean, Hard Voting)
  - Detailed metrics: accuracy, confusion matrix, per-class metrics
  - Model size and inference speed analysis
- **Automatic Data Download**: Fashion-MNIST dataset is automatically downloaded on first run
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **GPU Support**: Automatic CUDA detection and usage when available

## 🧠 Models

| Model | Architecture | Input Size | Parameters | Description |
|-------|-------------|------------|------------|-------------|
| **BetterCNN** | Custom CNN | 28×28×1 | ~2M | Lightweight CNN designed specifically for small images |
| **ResNet18** | ResNet-18 | 28×28×1 | ~11M | Residual network with skip connections |
| **EfficientNet-B0** | EfficientNet-B0 | 128×128×1 | ~5M | Efficient architecture with compound scaling |
| **DenseNet-121** | DenseNet-121 | 28×28×1 | ~8M | Densely connected convolutional network |
| **ViT-Tiny** | Vision Transformer | 224×224×3 | ~5M | Transformer-based architecture for vision tasks |

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Setup

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/macOS
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install PyTorch with CUDA support** (optional, for GPU acceleration):
   - Visit [PyTorch official website](https://pytorch.org/) to get the installation command for your CUDA version
   - Example for CUDA 12.1:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

## 📖 Usage

### Training Models

Each model can be trained independently:

```bash
# Train BetterCNN
python train_cnn.py

# Train ResNet18
python train_resnet.py

# Train EfficientNet-B0
python train_efficientnet.py

# Train DenseNet-121
python train_densenet.py

# Train ViT-Tiny
python train_vit.py
```

**Note**: The Fashion-MNIST dataset will be automatically downloaded to `./data/` on the first run.

### Evaluating Models

Evaluate all trained models with comprehensive metrics:

```bash
# Evaluate with TTA (Test-Time Augmentation) - default
python evaluate_all_models.py

# Evaluate without TTA
python evaluate_all_models.py no_tta
```

The evaluation script will:
- Load all available trained models
- Generate confusion matrices
- Calculate detailed metrics (accuracy, per-class accuracy, etc.)
- Measure inference speed
- Create ensemble models using multiple methods
- Save results to `evaluation_results/` directory

### Output Files

After evaluation, you'll find:
- `evaluation_results/all_models_results.json` - Detailed metrics in JSON format
- `evaluation_results/model_comparison.md` - Comparison table in Markdown
- `evaluation_results/confusion_matrix_*.png` - Confusion matrix visualizations for each model
- `evaluation_results/confusion_matrix_ensemble.png` - Ensemble model confusion matrix

## 📁 Project Structure

```
fashion-mnist/
├── train_cnn.py              # BetterCNN training script
├── train_resnet.py            # ResNet18 training script
├── train_efficientnet.py      # EfficientNet-B0 training script
├── train_densenet.py          # DenseNet-121 training script
├── train_vit.py               # ViT-Tiny training script
├── evaluate_all_models.py     # Unified evaluation script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/                      # Fashion-MNIST dataset (auto-downloaded)
│   └── FashionMNIST/
│       └── raw/
│
├── evaluation_results/        # Evaluation outputs (generated)
│   ├── all_models_results.json
│   ├── model_comparison.md
│   └── confusion_matrix_*.png
│
└── *.pt                       # Trained model weights (generated)
    ├── better_cnn_fmnist.pt
    ├── resnet_fmnist.pt
    ├── efficientnet_fmnist.pt
    ├── densenet_fmnist.pt
    └── vit_fmnist.pt
```

## 📦 Requirements

All dependencies are listed in `requirements.txt`:

- **torch** >= 2.0.0 - PyTorch deep learning framework
- **torchvision** >= 0.15.0 - Datasets and image transformations
- **timm** >= 0.9.0 - Vision Transformer models
- **tqdm** >= 4.65.0 - Progress bars
- **numpy** >= 1.24.0 - Numerical computing
- **pillow** >= 10.0.0 - Image processing
- **matplotlib** >= 3.7.0 - Plotting and visualization
- **scikit-learn** >= 1.3.0 - Evaluation metrics
- **seaborn** >= 0.12.0 - Statistical visualization
- **pandas** >= 1.2.0 - Data manipulation (seaborn dependency)

## 📊 Results

The evaluation script generates comprehensive results including:

- **Accuracy**: Overall and per-class classification accuracy
- **Confusion Matrix**: Visual representation of classification performance
- **Model Size**: Memory footprint of each model
- **Inference Speed**: FPS (Frames Per Second) and latency measurements
- **Top Errors**: Most common misclassification patterns
- **Ensemble Performance**: Results from multiple ensemble methods

Results are saved in both JSON (for programmatic access) and Markdown (for human reading) formats.

## 🔧 Training Features

Each training script includes:

- **Reproducibility**: Fixed random seeds for consistent results
- **Mixed Precision Training**: FP16 for faster training and lower memory usage
- **Data Augmentation**:
  - Random horizontal flip
  - Random rotation
  - Random affine transformations
  - MixUp and CutMix (for some models)
  - Random erasing
- **Learning Rate Scheduling**: Warmup + Cosine Annealing
- **Early Stopping**: Prevents overfitting with configurable patience
- **Gradient Clipping**: Stabilizes training for some models

## 🌐 Portability

This project is designed to be portable:

- ✅ **No hardcoded paths**: All paths are relative to the script location
- ✅ **Automatic data download**: Fashion-MNIST is downloaded automatically
- ✅ **Cross-platform**: Works on Windows, Linux, and macOS
- ✅ **Standalone scripts**: Each training script can run independently
- ✅ **Self-contained**: No external data files required (except auto-downloaded dataset)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- Fashion-MNIST dataset: [GitHub](https://github.com/zalandoresearch/fashion-mnist)
- PyTorch: [Official Website](https://pytorch.org/)
- timm library: [GitHub](https://github.com/huggingface/pytorch-image-models)

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Note**: This project is designed for educational purposes and demonstrates various deep learning techniques for image classification on the Fashion-MNIST dataset.

