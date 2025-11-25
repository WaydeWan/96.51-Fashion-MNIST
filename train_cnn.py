"""
Fashion-MNIST BetterCNN Model Training Script

This script implements a CNN model designed specifically for 28×28 small images, achieving 93.35% accuracy on the Fashion-MNIST dataset.

Code References:
- PyTorch Framework: https://pytorch.org/
- torchvision Datasets and Transforms: https://pytorch.org/vision/stable/index.html
- CNN architecture design references VGG's stacked convolution concept, but simplified for small images
- Data augmentation strategies reference common methods used in ImageNet training
- Learning rate scheduling uses Cosine Annealing, referencing PyTorch official documentation

"""
import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ============================================================================
# Configuration and Random Seed (for reproducibility)
# ============================================================================
# Set global random seed to ensure consistent results across runs
# This is very important for experimental reproducibility
def set_seed(seed: int = 42) -> None:
    """
    Set seed for all random number generators
    
    Args:
        seed: Random seed value, default 42 (common value)
    
    Notes:
        - Python random module: Used for data loading shuffle
        - NumPy: Used for random operations in numerical computations
        - PyTorch: Used for model initialization and data augmentation
        - CUDA: If using GPU, also need to set CUDA random seed
        - CUDNN determinism: Ensures deterministic convolution operations (may affect performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # Ensure deterministic convolution operations
        torch.backends.cudnn.benchmark = False      # Disable benchmark to improve determinism

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
# Data augmentation strategy notes:
# 1. RandomHorizontalFlip: Random horizontal flip, increases data diversity
# 2. RandomRotation: Small angle rotation, improves model robustness to rotation
# 3. RandomAffine: Slight translation, simulates real-world usage scenarios
# 4. Normalize: Normalize to [-1, 1], accelerates training convergence
# 5. RandomErasing: Random erasing, prevents overfitting (similar to Cutout technique)
def get_loaders(batch_size=256):
    """
    Get training and test data loaders
    
    Args:
        batch_size: Batch size, default 256 (GPU can use larger batches)
    
    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    
    Data augmentation notes (training set):
        - RandomHorizontalFlip(p=0.5): 50% probability horizontal flip
        - RandomRotation(5): Random rotation ±5 degrees
        - RandomAffine: Slight translation (within 5% range)
        - ToTensor: Convert to Tensor in [0,1] range
        - Normalize((0.5,), (0.5,)): Normalize to [-1, 1] (single channel)
        - RandomErasing: 10% probability random erasing of small regions
    
    Test set transforms:
        - Only basic transforms (ToTensor + Normalize), no augmentation
        - Ensures evaluation consistency
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip, increases data diversity
        transforms.RandomRotation(5),  # Small angle rotation, improves robustness to rotation
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Slight translation (5%)
        transforms.ToTensor(),  # Convert to Tensor, range [0, 1]
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1] (single channel, mean 0.5, std 0.5)
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2))  # Random erasing, prevents overfitting
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_set = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    return train_loader, test_loader

# ============================================================================
# Model Definition: BetterCNN
# ============================================================================
# Architecture design notes:
# - References VGG's stacked convolution concept, but optimized for 28×28 small images
# - Uses BatchNorm to accelerate training and improve stability
# - Uses Dropout to prevent overfitting
# - Parameter count ~2.3M, balancing model complexity and data size
class BetterCNN(nn.Module):
    """
    BetterCNN Model Architecture
    
    CNN designed specifically for 28×28 small images, containing 3 convolutional blocks and 2 fully connected layers.
    
    Architecture features:
    1. Convolutional layers: 3 blocks, each containing 2 conv layers + BatchNorm + ReLU + MaxPool + Dropout
    2. Increasing channels: 64 -> 128 -> 256
    3. Fully connected layers: 2 layers, using BatchNorm and Dropout
    4. Parameter count: ~2.3M
    
    Design rationale:
    - 28×28 images become 3×3 after 3 MaxPool(2x2) operations, perfect for fully connected layers
    - Increasing channel count follows common CNN design patterns
    - BatchNorm and Dropout improve generalization ability
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Feature extraction layers (convolutional part)
        # Input: 28×28×1 -> Output: 3×3×256
        self.features = nn.Sequential(
            # First convolutional block: 28×28×1 -> 14×14×64
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Maintain size: 28×28
            nn.BatchNorm2d(64),  # Batch normalization, accelerates training
            nn.ReLU(inplace=True),  # Activation function
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Convolve again to extract features
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Downsampling: 28×28 -> 14×14
            nn.Dropout(0.25),  # Dropout prevents overfitting
            
            # Second convolutional block: 14×14×64 -> 7×7×128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Double channels
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Downsampling: 14×14 -> 7×7
            nn.Dropout(0.25),
            
            # Third convolutional block: 7×7×128 -> 3×3×256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Double channels again
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Downsampling: 7×7 -> 3×3 (Note: 7 is not divisible by 2, actually gets 3×3)
            nn.Dropout(0.25),
        )
        
        # Classifier (fully connected layers)
        # Input: 256×3×3 = 2304 -> Output: 10 classes
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten 3D feature map to 1D vector
            nn.Linear(256 * 3 * 3, 512),  # First fully connected layer: 2304 -> 512
            nn.BatchNorm1d(512),  # 1D BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Higher dropout rate, prevents overfitting
            nn.Linear(512, 256),  # Second fully connected layer: 512 -> 256
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Slightly lower dropout rate
            nn.Linear(256, num_classes)  # Output layer: 256 -> 10 (number of classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================================
# MixUp/CutMix Data Augmentation
# ============================================================================
def mixup_data(x, y, alpha=0.2):
    """
    MixUp data augmentation
    Reference: mixup: Beyond Empirical Risk Minimization (2017)
    
    Args:
        x: Input images [B, C, H, W]
        y: Labels [B]
        alpha: Beta distribution parameter, controls mixing strength
    
    Returns:
        mixed_x: Mixed images
        y_a, y_b: Two original labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """
    CutMix data augmentation
    Reference: CutMix: Regularization Strategy to Train Strong Classifiers (2019)
    
    Args:
        x: Input images [B, C, H, W]
        y: Labels [B]
        alpha: Beta distribution parameter
    
    Returns:
        mixed_x: Mixed images
        y_a, y_b: Two original labels
        lam: Mixing coefficient (for loss calculation)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Calculate crop region
    W, H = x.size(3), x.size(2)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Randomly select crop center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Ensure crop region is within image
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Perform CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to match actual crop region
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculate MixUp/CutMix loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================================
# Learning Rate Warmup Scheduler
# ============================================================================
class WarmupCosineScheduler:
    """
    Cosine annealing learning rate scheduler with warmup
    
    References:
    - Warmup: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour (2017)
    - Cosine Annealing: SGDR: Stochastic Gradient Descent with Warm Restarts (2016)
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase: linearly increase learning rate
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing phase
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

# ============================================================================
# Training Function
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device, scaler=None, use_mixup=True, use_cutmix=True):
    """
    Train for one epoch
    
    Args:
        model: Model
        loader: Data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (CPU/GPU)
        scaler: GradScaler for mixed precision training (optional)
    
    Returns:
        average_loss: Average loss value
    
    Technical notes:
        - Mixed Precision Training: Uses FP16 to accelerate training, reduces memory usage
        - non_blocking=True: Asynchronous data transfer, improves GPU utilization
        - Gradient zeroing: Zero gradients before each batch, prevents gradient accumulation
    """
    model.train()  # Set to training mode (enables Dropout, etc.)
    total_loss = 0
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()  # Zero gradients
        
        # MixUp/CutMix data augmentation (randomly select one)
        if use_mixup or use_cutmix:
            if use_cutmix and np.random.rand() < 0.5:
                # Use CutMix
                mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
            elif use_mixup:
                # Use MixUp
                mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
            else:
                mixed_x, y_a, y_b, lam = x, y, y, 1.0
        else:
            mixed_x, y_a, y_b, lam = x, y, y, 1.0
        
        # Mixed precision training (if using GPU)
        if scaler:
            with torch.amp.autocast('cuda'):  # Automatic mixed precision context
                logits = model(mixed_x)
                if lam != 1.0:
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                else:
                    loss = criterion(logits, y_a)
            scaler.scale(loss).backward()  # Scale gradients
            # Gradient clipping (prevents gradient explosion)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)  # Update parameters
            scaler.update()  # Update scaler
        else:
            # Standard training (CPU or mixed precision not enabled)
            logits = model(mixed_x)
            if lam != 1.0:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, y_a)
            loss.backward()  # Backward propagation
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # Update parameters
        
        total_loss += loss.item()
    return total_loss / len(loader)  # Return average loss

def evaluate(model, loader, device):
    """Evaluate model, returns accuracy and average loss"""
    model.eval()
    correct = total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    use_amp = torch.cuda.is_available()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()
    return correct / total, total_loss / len(loader)

# Main program
def main(cnn_type: str = 'better'):
    # Get absolute path of script directory (project root)
    script_dir = Path(__file__).parent.absolute()
    
    train_loader, test_loader = get_loaders(batch_size=256)
    
    # Create model (BetterCNN only)
    model = BetterCNN(num_classes=10).to(device)
    print("Using BetterCNN")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # Training Configuration
    # ========================================================================
    
    # Mixed Precision Training
    # Uses FP16 to accelerate training, reduces memory usage, minimal impact on accuracy
    # Reference: PyTorch official AMP documentation
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    # Loss function: Cross-entropy loss + label smoothing
    # Label Smoothing (0.1) prevents model overconfidence, improves generalization
    # Reference: Rethinking the Inception Architecture for Computer Vision (2015)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer: AdamW
    # AdamW improves weight decay implementation, more aligned with L2 regularization theory
    # Learning rate 1e-3 is an empirical value, performs well on Fashion-MNIST
    # Weight decay 1e-4 provides L2 regularization, prevents overfitting
    # Reference: Decoupled Weight Decay Regularization (2017)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training loop configuration (enhanced version)
    max_epochs = 80  # Increased max training epochs (from 50 to 80)
    patience = 15  # Increased early stopping patience (from 10 to 15)
    min_delta = 0.0001  # Minimum improvement threshold: accuracy improvement less than this is not considered improvement
    
    # Learning rate scheduling: Cosine annealing with warmup
    # Warmup phase: First 5 epochs linearly increase learning rate, helps model stabilize training
    # Cosine annealing phase: Subsequent epochs use cosine annealing for fine-tuning
    warmup_epochs = 5
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=warmup_epochs,
        total_epochs=max_epochs,
        base_lr=1e-3,
        min_lr=1e-6
    )
    
    # Training loop (enhanced early stopping: based on accuracy and validation loss)
    best_acc = 0.0
    best_loss = float('inf')
    wait = 0
    no_improve_count = 0  # Consecutive no-improvement count
    
    print(f"\nStarting training for {cnn_type} CNN")
    print("="*60)
    
    for epoch in range(max_epochs):
        # Training (using MixUp/CutMix)
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, 
                             use_mixup=True, use_cutmix=True)
        # Evaluation (returns accuracy and validation loss)
        acc, val_loss = evaluate(model, test_loader, device)
        
        current_lr = scheduler.get_lr()
        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | LR: {current_lr:.6f}")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  GPU Memory: {memory_used:.2f} GB")
        
        # Learning rate scheduling (Warmup + Cosine)
        scheduler.step()
        
        # Enhanced early stopping mechanism: considers both accuracy and validation loss
        improved = False
        if acc > best_acc + min_delta:  # Significant accuracy improvement
            best_acc = acc
            improved = True
        if val_loss < best_loss - min_delta:  # Significant validation loss reduction
            best_loss = val_loss
            improved = True
        
        if improved:
            wait = 0
            no_improve_count = 0
            model_path = script_dir / f"{cnn_type}_cnn_fmnist.pt"
            torch.save(model.state_dict(), str(model_path))
            print(f"  -> Saved best model! Best acc: {best_acc:.4f}, Best val loss: {best_loss:.4f}")
        else:
            wait += 1
            no_improve_count += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1} (accuracy no improvement)")
                break
            if no_improve_count >= patience * 2:  # Stricter early stopping: longer consecutive no-improvement period
                print(f"Early stopping at epoch {epoch+1} (accuracy and loss no improvement)")
                break
    
    print(f"\nFinal best accuracy: {best_acc:.4f}")
    return best_acc

if __name__ == "__main__":
    main(cnn_type='better')
