"""
Fashion-MNIST ResNet18 训练脚本（优化版）
- 增强数据增强策略
- 添加梯度裁剪
- 优化学习率调度
- 添加随机种子保证可复现
"""
import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# 设置随机种子（保证可复现）
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 数据加载（增强版）
def get_loaders(batch_size=256):
    # 训练集：更强的数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),  # 小角度旋转
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # 轻微平移
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2))  # 随机擦除
    ])
    
    # 测试集：只做基本转换
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
    
    # Windows上num_workers=0避免多进程问题
    num_workers = 0
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader

# 模型：轻量ResNet18
class ResNet28(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 使用预训练ResNet18（虽然没有28x28的预训练，但结构是好的）
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # 修改第一层适应单通道
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # 移除maxpool以适应28x28
        
        # 修改分类头
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.model(x)

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

# 训练函数（支持混合精度 + 梯度裁剪 + MixUp/CutMix）
def train_epoch(model, loader, criterion, optimizer, device, scaler=None, max_grad_norm=1.0, use_mixup=True, use_cutmix=True):
    model.train()
    total_loss = 0
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
        
        # 混合精度训练（GPU加速）
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(mixed_x)
                if lam != 1.0:
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                else:
                    loss = criterion(logits, y_a)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(mixed_x)
            if lam != 1.0:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, y_a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    """评估模型，返回准确率和平均损失"""
    model.eval()
    correct = total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    use_amp = torch.cuda.is_available()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
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

# 主程序
def main():
    # 获取脚本所在目录（项目根目录）的绝对路径
    script_dir = Path(__file__).parent.absolute()
    # 加载数据
    train_loader, test_loader = get_loaders(batch_size=256)
    
    # 创建模型
    model = ResNet28(num_classes=10).to(device)
    
    # 混合精度训练（GPU加速）
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 学习率调度：带Warmup的余弦退火
    max_epochs = 80
    warmup_epochs = 5
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=warmup_epochs,
        total_epochs=max_epochs,
        base_lr=1e-3,
        min_lr=1e-6
    )
    
    # 训练循环（增强早停）
    best_acc = 0.0
    best_loss = float('inf')
    patience = 15
    min_delta = 0.0001
    wait = 0
    no_improve_count = 0
    
    print("\n开始训练 ResNet18")
    print("="*60)
    
    for epoch in range(max_epochs):
        # 训练（使用MixUp/CutMix）
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, 
                               max_grad_norm=1.0, use_mixup=True, use_cutmix=True)
        
        # 评估
        acc, val_loss = evaluate(model, test_loader, device)
        
        current_lr = scheduler.get_lr()
        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | LR: {current_lr:.6f}")
        
        # GPU内存监控
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  GPU Memory: {memory_used:.2f} GB")
        
        # 学习率调度
        scheduler.step()
        
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
            model_path = script_dir / "resnet_fmnist.pt"
            torch.save(model.state_dict(), str(model_path))
            print(f"  -> 保存最佳模型! Best acc: {best_acc:.4f}, Best val loss: {best_loss:.4f}")
        else:
            wait += 1
            no_improve_count += 1
            if wait >= patience:
                print(f"早停于 epoch {epoch+1} (准确率无改善)")
                break
            if no_improve_count >= patience * 2:
                print(f"早停于 epoch {epoch+1} (准确率和损失均无改善)")
                break
    
    print(f"\nFinal Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
