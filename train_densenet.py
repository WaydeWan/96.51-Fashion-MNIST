"""
Fashion-MNIST DenseNet 训练脚本

基于 torchvision 的 DenseNet-121，适配28×28灰度输入。
"""

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import DenseNet121_Weights, densenet121
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# MixUp/CutMix 数据增强
# ============================================================================
def mixup_data(x, y, alpha=0.2):
    """MixUp数据增强"""
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
    """CutMix数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    W, H = x.size(3), x.size(2)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """计算MixUp/CutMix的损失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================================
# 学习率Warmup调度器
# ============================================================================
class WarmupCosineScheduler:
    """带Warmup的余弦退火学习率调度器"""
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
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def get_loaders(batch_size: int = 192) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=test_transform)

    # Windows上num_workers=0避免多进程问题，Linux/Mac可以用4
    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def build_model(num_classes: int = 10) -> nn.Module:
    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = densenet121(weights=weights)

    # 修改首层卷积，适配单通道及小尺寸
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.features.pool0 = nn.Identity()

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, scaler=None, use_mixup=True, use_cutmix=True):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()

        # MixUp/CutMix数据增强
        if use_mixup or use_cutmix:
            if use_cutmix and np.random.rand() < 0.5:
                mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
            elif use_mixup:
                mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
            else:
                mixed_x, y_a, y_b, lam = x, y, y, 1.0
        else:
            mixed_x, y_a, y_b, lam = x, y, y, 1.0

        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(mixed_x)
                if lam != 1.0:
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                else:
                    loss = criterion(logits, y_a)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(mixed_x)
            if lam != 1.0:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, y_a)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    """评估模型，返回准确率和平均损失"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()
    return correct / total, total_loss / len(loader)


def main():
    # 获取脚本所在目录（项目根目录）的绝对路径
    script_dir = Path(__file__).parent.absolute()
    
    train_loader, test_loader = get_loaders(batch_size=192)
    model = build_model(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 学习率调度：带Warmup的余弦退火
    max_epochs = 80
    warmup_epochs = 5
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=warmup_epochs,
        total_epochs=max_epochs,
        base_lr=1e-4,
        min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_acc = 0.0
    best_loss = float('inf')
    patience = 15
    min_delta = 0.0001
    wait = 0
    no_improve_count = 0

    for epoch in range(max_epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, 
                                  use_mixup=True, use_cutmix=True)
        acc, val_loss = evaluate(model, test_loader)
        scheduler.step()

        current_lr = scheduler.get_lr()
        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | LR: {current_lr:.6f}")

        # 增强早停机制
        improved = False
        if acc > best_acc + min_delta:
            best_acc = acc
            improved = True
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            improved = True

        if improved:
            wait = 0
            no_improve_count = 0
            model_path = script_dir / "densenet_fmnist.pt"
            torch.save(model.state_dict(), str(model_path))
            print(f"  -> Save best model: {best_acc:.4f}, Best val loss: {best_loss:.4f}")
        else:
            wait += 1
            no_improve_count += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1} (accuracy no improvement)")
                break
            if no_improve_count >= patience * 2:
                print(f"Early stopping at epoch {epoch+1} (accuracy and loss no improvement)")
                break

    print(f"Best Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()

