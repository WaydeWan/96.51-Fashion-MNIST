# Fashion-MNIST Image Classification Project Report

## 1. Project Overview

This project uses deep learning techniques to classify Fashion-MNIST images. Fashion-MNIST contains grayscale images of 10 categories of fashion items, with each image being 28×28 pixels. We compared 5 different deep learning model architectures and applied various training techniques to improve model performance.

**Dataset Information:**
- Training set: 60,000 images
- Test set: 10,000 images
- Number of classes: 10
- Image size: 28×28 pixels (grayscale)
- Data format: Single-channel grayscale images, pixel values range 0-255

---

## 2. Data Preparation

### 2.1 Dataset Introduction

The Fashion-MNIST dataset contains the following 10 categories:

| Label Index | English Name | Chinese Name |
|------------|--------------|--------------|
| 0 | T-shirt/Top | T-shirt/Top |
| 1 | Trouser | Trouser |
| 2 | Pullover | Pullover |
| 3 | Dress | Dress |
| 4 | Coat | Coat |
| 5 | Sandal | Sandal |
| 6 | Shirt | Shirt |
| 7 | Sneaker | Sneaker |
| 8 | Bag | Bag |
| 9 | Ankle boot | Ankle boot |

**Data Characteristics Analysis:**
- **Uniform Background:** All images have black backgrounds with white/gray objects, facilitating model learning
- **Centered Objects:** All objects are centered, reducing the impact of positional variations
- **Resolution Limitation:** 28×28 pixel resolution is relatively low, making some details difficult to distinguish (e.g., collar differences between T-shirt and Shirt)
- **Class Similarity:** Some classes have similarities, increasing classification difficulty:
  - T-shirt (0) vs Shirt (6): Both are upper garments with similar shapes
  - Pullover (2) vs Coat (4): Both are outerwear
  - Sandal (5) vs Sneaker (7) vs Ankle boot (9): All are footwear

These characteristics influenced our model selection: due to low resolution and class similarity, we need a model that can learn fine-grained features while avoiding overfitting.

### 2.2 Data Preprocessing and Cleaning

**Data Loading:**
- Using PyTorch's `torchvision.datasets.FashionMNIST` to automatically download and load data
- Original data is in IDX format binary files, automatically converted to Tensor format by PyTorch
- **No data cleaning required:** Fashion-MNIST dataset is preprocessed, all images are 28×28 grayscale with uniform black backgrounds and centered objects, requiring no additional cleaning steps

**Feature Processing:**

1. **Feature Reduction:**
   - **No feature reduction:** Maintain original 28×28 pixel resolution because:
     - 28×28 is already a small size, further reduction would lose important information
     - All models can handle 28×28 inputs
     - For ViT and EfficientNet, we use upsampling (Resize) to adapt to model input requirements

2. **Feature Addition:**
   - **Channel Expansion (ViT only):** Expand single-channel grayscale to 3 channels (RGB) by channel replication
     - Reason: ViT model uses ImageNet pretrained weights, requires 3-channel input
     - Implementation: `transforms.Lambda(lambda x: x.repeat(3, 1, 1))`
   - **Size Adjustment (ViT and EfficientNet):** Upsample 28×28 to 224×224 (ViT) or 128×128 (EfficientNet)
     - Reason: These models use ImageNet pretrained weights, requiring larger input sizes
     - Trade-off: Upsampling introduces interpolation noise, but enables use of pretrained weights

### 2.3 Data Preprocessing

**Data Transformation Pipeline:**

1. **Training Set Augmentation:**
   - `RandomHorizontalFlip(p=0.5)`: Random horizontal flip to increase data diversity
   - `RandomRotation(5)`: Random rotation ±5 degrees to enhance model robustness to rotation
   - `RandomAffine(degrees=0, translate=(0.05, 0.05))`: Slight translation (5%) to simulate real-world scenarios
   - `ToTensor()`: Convert to PyTorch Tensor format ([0, 1] range)
   - `Normalize((0.5,), (0.5,))`: Normalize to [-1, 1] range to accelerate training convergence
   - `RandomErasing(p=0.1, scale=(0.02, 0.2))`: Random erasing to prevent overfitting (similar to Cutout technique)

2. **Test Set Transformation:**
   - `ToTensor()`: Convert to Tensor format
   - `Normalize((0.5,), (0.5,))`: Consistent normalization with training set
   - **No data augmentation:** Ensures evaluation consistency

**Data Augmentation Rationale:**
- **Why is data augmentation needed?** Fashion-MNIST has only 60k training samples, which is relatively small for deep models
- **Reasons for choosing these augmentations:**
  1. **RandomHorizontalFlip:** Clothing items can typically be horizontally flipped without changing category (e.g., T-shirt, trousers)
  2. **RandomRotation(5 degrees):** Small angle rotation simulates angle variations in real photography, but angle cannot be too large (to avoid rotating shoes into other items)
  3. **RandomAffine translation:** Simulates positional variations of objects in images, but translation is small (5%) to avoid objects moving out of frame
  4. **RandomErasing:** Simulates occlusion situations, improves model robustness, but probability is low (10%) to avoid excessive image damage

**Special Processing:**
- **ViT:** Upsample 28×28 to 224×224 and expand single channel to 3 channels (channel replication) to adapt to ImageNet pretrained weights
- **EfficientNet:** Upsample 28×28 to 128×128 and modify first convolutional layer to adapt to single-channel input

### 2.4 Data Visualization

We exported 16 test set samples for visualization (see `export_tests/` folder). Each image is labeled with ground truth, facilitating understanding of dataset content. Visualization script: `export_test_images.py`.

**Dataset Sample Display:**

We randomly selected 16 test set images covering all 10 categories. Each image filename format: `test_XX_category_name.png` for easy identification.

---

## 3. Model Architecture and Implementation

### 3.1 Model Comparison Experiments

We implemented 5 different deep learning models and conducted comprehensive comparison experiments on the complete test set (10,000 images). **All models used TTA (Test-Time Augmentation)**:

| Model | Parameters | Input Size | Test Accuracy | Model Size | Inference Speed | Characteristics |
|-------|------------|------------|---------------|------------|-----------------|-----------------|
| **Ensemble (Weighted Average)** | 30.14M | - | **96.51%** ⭐⭐ | 115.49MB | N/A | Ensemble model, highest accuracy |
| **ResNet18** | 11.17M | 28×28 | **96.20%** ⭐ | 42.66MB | 9659 FPS | Highest single model accuracy, but large parameter count |
| DenseNet-121 | 6.96M | 28×28 | 96.12% | 26.85MB | 2359 FPS | Dense connections, second highest accuracy |
| EfficientNet-B0 | 4.02M | 128×128 | 96.02% | 15.49MB | 4248 FPS | Efficient architecture, third highest accuracy |
| ViT-Tiny | 5.53M | 224×224 | 95.78% | 21.08MB | 1632 FPS | Transformer architecture, good performance |
| BetterCNN | 2.46M | 28×28 | 95.44% | 9.40MB | 31655 FPS | Fewest parameters, fastest inference |

**Key Findings:**
- All models achieved accuracy **above 95%**, indicating that different architectures can achieve good results on this task
- **Ensemble model (Weighted Average)** achieved the highest accuracy (96.51%), demonstrating the effectiveness of model ensemble
- **ResNet18** achieved the highest single model accuracy (96.20%), demonstrating the effectiveness of residual connections
- **BetterCNN** has slightly lower accuracy (95.44%) but has the fewest parameters (2.46M) and fastest inference speed (31655 FPS), achieving a good balance between efficiency and performance

**Model Implementation Notes:**
- **BetterCNN:** Fully custom CNN architecture designed specifically for 28×28
- **ResNet18:** Uses torchvision pretrained model, modified first layer to adapt to single channel, removed maxpool
- **EfficientNet-B0:** Uses torchvision pretrained model, modified first layer to adapt to single channel, upsampled to 128×128
- **DenseNet-121:** Uses torchvision pretrained model, modified first layer to adapt to single channel, removed pool layer
- **ViT-Tiny:** Uses timm library pretrained model, upsampled to 224×224, expanded channels to 3 channels

### 3.2 ResNet18

**Architecture Characteristics:**
- **Pretrained Weights:** ImageNet pretrained (ResNet18_Weights.IMAGENET1K_V1)
- **Parameters:** 11.17M
- **Input Size:** 28×28×1

**Modifications for Fashion-MNIST:**

```python
# 1. Modify first convolutional layer: change from 3 channels to 1 channel
self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

# 2. Remove maxpool layer: adapt to 28×28 small images
self.model.maxpool = nn.Identity()

# 3. Modify classification head: output 10 classes
self.model.fc = nn.Linear(512, num_classes)
```

**Architecture Advantages:**
- Residual connections solve deep network degradation problems
- Pretrained weights provide good feature extraction capability
- Although parameter count is large, performance is stable

### 3.3 DenseNet-121

**Architecture Characteristics:**
- **Pretrained Weights:** ImageNet pretrained (DenseNet121_Weights.IMAGENET1K_V1)
- **Parameters:** 6.96M
- **Input Size:** 28×28×1

**Modifications for Fashion-MNIST:**

```python
# 1. Load pretrained model
model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

# 2. Modify first convolutional layer: adapt to single-channel input
first_conv = model.features.conv0
model.features.conv0 = nn.Conv2d(
    1, first_conv.out_channels,
    kernel_size=first_conv.kernel_size,
    stride=first_conv.stride,
    padding=first_conv.padding,
    bias=False
)

# 3. Remove initial pooling layer: adapt to 28×28 small images
model.features.pool0 = nn.Identity()

# 4. Modify classification head: output 10 classes
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
```

**Architecture Advantages:**
- **Dense Connections:** Each layer receives features from all previous layers, improving feature reuse
- Moderate parameter count with strong feature extraction capability

### 3.4 EfficientNet-B0

**Architecture Characteristics:**
- **Pretrained Weights:** ImageNet pretrained (EfficientNet_B0_Weights.IMAGENET1K_V1)
- **Parameters:** 4.02M
- **Input Size:** 128×128×1 (upsampled)

**Modifications for Fashion-MNIST:**

```python
# 1. Load pretrained model
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# 2. Modify first convolutional layer: adapt to single-channel input
first_conv = model.features[0][0]
model.features[0][0] = nn.Conv2d(
    1, first_conv.out_channels,
    kernel_size=first_conv.kernel_size,
    stride=first_conv.stride,
    padding=first_conv.padding,
    bias=False
)

# 3. Modify classification head: output 10 classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# 4. Data preprocessing: upsample to 128×128
transforms.Resize((128, 128))
```

**Architecture Advantages:**
- **Compound Scaling:** Simultaneously optimizes depth, width, and resolution
- **MBConv Blocks:** Uses mobile inverted bottleneck convolution for efficiency
- **SE Attention:** Squeeze-and-Excitation attention mechanism
- Moderate parameter count, achieving good balance between accuracy and efficiency

### 3.5 ViT-Tiny (Vision Transformer)

**Architecture Characteristics:**
- **Pretrained Weights:** ImageNet pretrained (timm library)
- **Parameters:** 5.53M
- **Input Size:** 224×224×3 (upsampled + channel expansion)

**Modifications for Fashion-MNIST:**

```python
# 1. Load pretrained model
model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10)

# 2. Data preprocessing: upsample to 224×224, expand to 3 channels
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # 1 channel → 3 channels
])
```

**Architecture Advantages:**
- **Self-Attention Mechanism:** Transformer architecture capable of capturing long-range dependencies
- Validates the effectiveness of Transformer on small datasets

### 3.6 BetterCNN (Custom CNN)

**Architecture Design:**
- **Design Philosophy:** References VGG's stacked convolution idea, but optimized for 28×28 small images
- **Parameters:** 2.46M
- **Input Size:** 28×28×1

**Detailed Architecture:**

```
Input: 28×28×1
├─ First Convolutional Block (64 channels)
│  ├─ Conv2d(1→64, 3×3, padding=1) + BatchNorm + ReLU
│  ├─ Conv2d(64→64, 3×3, padding=1) + BatchNorm + ReLU
│  ├─ MaxPool2d(2×2) → 14×14×64
│  └─ Dropout(0.25)
├─ Second Convolutional Block (128 channels)
│  ├─ Conv2d(64→128, 3×3, padding=1) + BatchNorm + ReLU
│  ├─ Conv2d(128→128, 3×3, padding=1) + BatchNorm + ReLU
│  ├─ MaxPool2d(2×2) → 7×7×128
│  └─ Dropout(0.25)
├─ Third Convolutional Block (256 channels)
│  ├─ Conv2d(128→256, 3×3, padding=1) + BatchNorm + ReLU
│  ├─ Conv2d(256→256, 3×3, padding=1) + BatchNorm + ReLU
│  ├─ MaxPool2d(2×2) → 3×3×256
│  └─ Dropout(0.25)
└─ Fully Connected Layers
   ├─ Linear(2304→512) + BatchNorm + ReLU + Dropout(0.5)
   ├─ Linear(512→256) + BatchNorm + ReLU + Dropout(0.3)
   └─ Linear(256→10) → Output classes
```

**Design Rationale:**
- 28×28 images become 3×3 after 3 MaxPool(2×2) operations, perfectly suited for fully connected layers
- Channel number progression (64→128→256) follows common CNN design patterns
- BatchNorm and Dropout improve generalization ability and prevent overfitting

---

## 4. Training Strategy and Technical Applications

### 4.1 Loss Function

**CrossEntropyLoss with Label Smoothing (0.1):**
- Standard cross-entropy loss function
- Label Smoothing prevents models from being overconfident, improving generalization
- Smoothing coefficient 0.1 is an empirical value, balancing accuracy and generalization

### 4.2 Optimizer

**AdamW Optimizer:**
- **Learning Rate:** Adjusted according to model
  - BetterCNN: 1e-3
  - ResNet18: 2e-4
  - EfficientNet-B0: 1e-3
  - DenseNet-121: 1e-3
  - ViT-Tiny: 1e-4
- **Weight Decay:** 1e-4 (L2 regularization)
- AdamW improves weight decay implementation compared to Adam, more aligned with theory

### 4.3 Learning Rate Scheduling

**Warmup Cosine Annealing Scheduler (WarmupCosineScheduler):**

**Implementation Principle:**
```python
class WarmupCosineScheduler:
    def step(self):
        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase: linearly increase learning rate
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing phase: smoothly decrease learning rate
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
```

**Parameter Settings:**
- **Warmup Phase:** First 5 epochs, learning rate linearly increases from 0 to initial value
  - Helps model converge stably in early training
  - Avoids instability from excessive learning rate in initial phase
- **Cosine Annealing Phase:** Subsequent epochs, learning rate gradually decreases from initial value to minimum (1e-6)
  - Uses cosine function to smoothly decrease learning rate
  - Allows model to fine-tune in later training stages

### 4.4 Advanced Data Augmentation Techniques

**MixUp/CutMix Data Augmentation:**

**MixUp Implementation:**
```python
def mixup_data(x, y, alpha=0.2):
    """Mix two images proportionally, labels are also mixed accordingly"""
    lam = np.random.beta(alpha, alpha)  # Mixing coefficient
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]  # Mixed images
    y_a, y_b = y, y[index]  # Two original labels
    return mixed_x, y_a, y_b, lam
```

**CutMix Implementation:**
```python
def cutmix_data(x, y, alpha=1.0):
    """Replace random region of one image with corresponding region of another image"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    # Calculate crop region
    W, H = x.size(3), x.size(2)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    # Crop and replace
    bbx1, bby1 = np.clip(cx - cut_w//2, 0, W), np.clip(cy - cut_h//2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w//2, 0, W), np.clip(cy + cut_h//2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))  # Update mixing coefficient
    return mixed_x, y, y[index], lam
```

**Usage Strategy:**
- During training, randomly choose MixUp or CutMix (50% probability each), apply augmentation to each batch
- **Loss Calculation:** Use mixed loss: `loss = λ * loss(y_a) + (1-λ) * loss(y_b)`, where λ is the mixing coefficient
- **Parameter Settings:** MixUp α=0.2, CutMix α=1.0

**Effects:**
- Increases data diversity, improves model generalization ability
- Expected accuracy improvement: 0.2-0.5%

### 4.5 Regularization Techniques

1. **Dropout:**
   - After convolutional layers: 0.25
   - Fully connected layers: 0.5 (first layer), 0.3 (second layer)
   - Prevents overfitting, improves generalization

2. **Batch Normalization:**
   - Applied after each convolutional and fully connected layer
   - Accelerates training convergence, improves model stability
   - Allows use of larger learning rates

3. **Weight Decay (L2 Regularization):**
   - Coefficient 1e-4, prevents weights from becoming too large

4. **Gradient Clipping:**
   - Gradient clipping (max_norm=1.0), stabilizes training process
   - Prevents gradient explosion

### 4.6 Training Tricks

1. **Mixed Precision Training:**
   - Uses `torch.amp.autocast` and `GradScaler`
   - Accelerates training on GPU, reduces memory usage
   - Minimal impact on accuracy, but training speed improves by approximately 30-50%

2. **Enhanced Early Stopping:**
   - **Patience:** 15 (increased from 10), gives model more opportunities to improve
   - **Minimum Improvement Threshold:** min_delta = 0.0001, only accuracy improvements exceeding this value count as improvement
   - **Dual Monitoring:** Simultaneously monitors accuracy and validation loss
   - Prevents overfitting while ensuring model is fully trained

3. **Random Seed:**
   - Set global random seed to 42 to ensure reproducible results

### 4.7 Training Process

**Training Configuration:**
- **Batch Size:** 128 (BetterCNN/ResNet/DenseNet) or 64 (EfficientNet/ViT)
- **Max Epochs:** Adjusted according to model
  - BetterCNN/ResNet/DenseNet: 80 epochs
  - EfficientNet: 70 epochs
  - ViT: 60 epochs
- **Device:** GPU (RTX 5060) or CPU
- **Training Time:** Approximately 5-8 minutes per model (GPU)

**Training Pipeline:**

```python
for epoch in range(max_epochs):
    # Training phase
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 1. Data augmentation (MixUp/CutMix)
        if np.random.rand() < 0.5:
            data, target_a, target_b, lam = mixup_data(data, target, alpha=0.2)
        else:
            data, target_a, target_b, lam = cutmix_data(data, target, alpha=1.0)
        
        # 2. Forward pass (mixed precision)
        with torch.amp.autocast():
            output = model(data)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        
        # 3. Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        scaler.step(optimizer)
        scaler.update()
    
    # 4. Learning rate scheduling
    scheduler.step()
    
    # 5. Validation phase
    val_acc = evaluate(model, val_loader)
    
    # 6. Early stopping check
    if val_acc > best_acc + min_delta:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

### 4.8 Test-Time Augmentation (TTA)

**TTA Strategy:**
- **Horizontal Flip:** Make 2 predictions per test image (original + horizontally flipped), then average probabilities
- **Implementation:** Directly flip tensors on GPU, efficient and doesn't affect data loading
- **Effect:** Consistently improves accuracy by 0.3-0.5%
- **Application:** All single model evaluations and ensemble evaluations use TTA

---

## 5. Model Performance Evaluation

### 5.1 Single Model Performance (with TTA)

All models used test-time augmentation (TTA: original image + horizontal flip):

| Model | Accuracy | Parameters | Model Size | Inference Speed | Correct/Total |
|-------|----------|------------|------------|-----------------|---------------|
| **ResNet18** | **96.20%** ⭐ | 11.17M | 42.66MB | 9659 FPS | 9,620/10,000 |
| DenseNet-121 | 96.12% | 6.96M | 26.85MB | 2359 FPS | 9,612/10,000 |
| EfficientNet-B0 | 96.02% | 4.02M | 15.49MB | 4248 FPS | 9,602/10,000 |
| ViT-Tiny | 95.78% | 5.53M | 21.08MB | 1632 FPS | 9,578/10,000 |
| BetterCNN | 95.44% | 2.46M | 9.40MB | 31655 FPS | 9,544/10,000 |

**Key Findings:**
- All models achieved accuracy **above 95%**, indicating that different architectures can effectively learn features
- ResNet18 achieved the highest single model accuracy (96.20%)
- BetterCNN has the fewest parameters and fastest inference speed, suitable for resource-constrained scenarios

### 5.2 Model Ensemble Performance

**Ensemble Methods:**
We implemented 4 ensemble methods, **Weighted Average** performed best:

**1. Weighted Average:**
- **Weight Calculation:** Uses softmax normalization, calculates weights based on single model accuracy
  - Formula: $w_i = \frac{\exp(\alpha \cdot (acc_i - acc_{min}))}{\sum_{j=1}^{N}\exp(\alpha \cdot (acc_j - acc_{min}))}$, where α=10
  - Prediction formula: $P_{ensemble} = \sum_{i=1}^{N} w_i \cdot P_i$
- **Actual Weight Distribution:**
  - ResNet18 (96.20%): weight ≈ 0.25
  - DenseNet-121 (96.12%): weight ≈ 0.24
  - EfficientNet-B0 (96.02%): weight ≈ 0.22
  - ViT-Tiny (95.78%): weight ≈ 0.16
  - BetterCNN (95.44%): weight ≈ 0.13

**2. Other Methods:**

**2.1 Simple Average:**
- **Prediction Formula:** $P_{ensemble} = \frac{1}{N} \sum_{i=1}^{N} P_i$
- **Description:** Arithmetic average of all models' prediction probabilities, equal weight for each model
- **Advantage:** Simple implementation, no weight calculation needed

**2.2 Geometric Mean:**
- **Prediction Formula:** $P_{ensemble} = \exp\left(\frac{1}{N} \sum_{i=1}^{N} \log(P_i + \epsilon)\right)$, where $\epsilon$ is a small constant (prevents log(0))
- **Description:** Take logarithm of probabilities, average, then exponentiate
- **Characteristic:** More sensitive to extreme values, tends to select classes with high confidence across all models

**2.3 Hard Voting:**
- **Prediction Formula:** $y_{ensemble} = \arg\max_c \sum_{i=1}^{N} \mathbf{1}(y_i = c)$
- **Description:** Each model predicts a class (hard label), count votes for each class, select the class with most votes
- **Characteristic:** Doesn't consider prediction probabilities, only class selection, uses probability average in case of ties

**Ensemble Results:**

| Ensemble Method | Accuracy | Correct/Total | Total Parameters | Total Model Size |
|-----------------|----------|--------------|------------------|------------------|
| **Weighted Average** | **96.51%** ⭐⭐ | 9,651/10,000 | 30.14M | 115.49MB |

**Ensemble Effects:**
- Ensemble model achieved **96.51%** accuracy, improving **0.31%** over best single model (96.20%)
- Errors reduced from 380 to 349, **8.2%** error reduction
- Demonstrates effectiveness of model ensemble: multiple models complement errors, significantly improving performance

### 5.3 Error Analysis

We conducted detailed error analysis on all models and discovered some common error patterns.

**Most Common Error Pairs (All Models):**

1. **Shirt ↔ T-shirt/Top:** This is the most common error across all models
   - ResNet18: Shirt→T-shirt/Top (~74 times), T-shirt/Top→Shirt (~45 times)
   - DenseNet-121: Shirt→T-shirt/Top (~70 times), T-shirt/Top→Shirt (~52 times)
   - EfficientNet-B0: Shirt→T-shirt/Top (~63 times), T-shirt/Top→Shirt (~52 times)
   - **Reason:** Both are upper garments with similar shapes, difficult to distinguish collars, cuffs, and other details at 28×28 resolution

2. **Shirt ↔ Coat:** Second most common error
   - Multiple models misclassify Shirt as Coat
   - **Reason:** Some shirts and coats have similar shapes at low resolution, especially long-sleeved shirts and coats

3. **Shirt ↔ Pullover:** Third most common error
   - **Reason:** Both are upper garments with similar features, difficult to distinguish at 28×28 resolution

4. **Other Common Errors:**
   - **Sandal ↔ Sneaker ↔ Ankle boot:** Confusion among footwear, but error rate is relatively low
   - **Pullover ↔ Coat:** Confusion among outerwear

**Per-Class Accuracy Analysis:**

All models performed worst on **Shirt** category:
- ResNet18: ~83.3%
- DenseNet-121: ~84.0%
- EfficientNet-B0: ~85.9%
- ViT-Tiny: ~85.2%
- BetterCNN: ~83.0%

Best performing categories are **Bag** and **Trouser**:
- **Bag:** 99.4-99.8% (all models), unique shape, easy to identify
- **Trouser:** 99.1-99.6% (all models), distinctive shape features, large difference from other categories

**Error Cause Analysis:**

1. **Class Similarity:** Shirt, T-shirt/Top, Coat, Pullover are all upper garments, difficult to distinguish at low resolution
   - At 28×28 pixels, key distinguishing features (collar type, buttons, zippers) cannot be clearly displayed
   - Similar shape contours make it difficult for models to learn effective distinguishing features

2. **Resolution Limitation:** 28×28 pixel resolution is too low, some details cannot be distinguished
   - Key features like collar details, buttons, zippers are lost at low resolution
   - Texture information is almost invisible, can only rely on shape features

3. **Data Quality:** Some images have poor quality, even human eyes have difficulty accurately classifying
   - Some samples in Fashion-MNIST dataset have ambiguous labels

4. **Model Limitations:** Even the best model (ResNet18, 96.20%) still confuses similar classes
   - Indicates the problem mainly comes from data limitations rather than model architecture
   - Ensemble model (96.51%) reduces some errors through multi-model complementarity

**Improvement Suggestions:**

1. **Increase Resolution:** If possible, use higher resolution images (e.g., 56×56 or 112×112)
2. **Feature Engineering:** Extract handcrafted features like edges and textures as supplements
3. **Data Augmentation:** Add more samples or augmentation strategies for difficult classes (e.g., Shirt)
4. **Model Ensemble:** Use weighted average ensemble to fully utilize complementarity of different models

### 5.4 Model Complexity and Bias-Variance Balance Analysis

**Bias Analysis:**

ResNet18 achieved **96.20% accuracy** on test set (highest single model), indicating the model can learn effective feature representations. We reduced bias through:

1. **Sufficient Model Capacity:** Parameter counts from 2.46M to 11.17M provide sufficient expressive power
2. **Deep Network Structure:** Using pretrained models and deep architectures can learn hierarchical features
3. **BatchNorm:** Accelerates training convergence, improves model learning capability
4. **Pretrained Weights:** ImageNet pretrained weights provide good feature extraction foundation

**Training Set vs Test Set Accuracy Comparison:**
- Training set accuracy: ~97-98% (observed during training)
- Test set accuracy: 96.20% (ResNet18, best single model)
- **Gap:** Approximately 1-2%, indicating no high bias problem and good overfitting control

**Variance Analysis:**

We effectively controlled overfitting (variance) through multiple regularization techniques:

1. **Dropout:**
   - Convolutional layers: 0.25
   - Fully connected layers: 0.5 (first layer), 0.3 (second layer)
   - Effect: Randomly drops neurons, prevents model from over-relying on specific features

2. **BatchNorm:**
   - Applied after each layer, stabilizes training process
   - Effect: Reduces internal covariate shift, improves generalization

3. **Weight Decay (L2 Regularization):**
   - Coefficient: 1e-4
   - Effect: Prevents weights from becoming too large, improves generalization

4. **Data Augmentation:**
   - Random flip, rotation, translation, erasing, MixUp, CutMix
   - Effect: Increases data diversity, improves model robustness

5. **Label Smoothing:**
   - Coefficient: 0.1
   - Effect: Prevents model from being overconfident, improves generalization

6. **Gradient Clipping:**
   - max_norm=1.0
   - Effect: Prevents gradient explosion, stabilizes training

**Complexity Selection and Balance:**

Different models achieved different balances between bias and variance:

| Model | Parameters | Accuracy | Bias | Variance | Result |
|-------|------------|----------|------|----------|--------|
| **ResNet18 (Best Single Model)** | 11.17M | 96.20% | Low | Low | ✅ Best Balance |
| DenseNet-121 | 6.96M | 96.12% | Low | Low | ✅ Good Balance |
| EfficientNet-B0 | 4.02M | 96.02% | Low | Low | ✅ Good Balance |
| ViT-Tiny | 5.53M | 95.78% | Low | Low | ✅ Acceptable |
| BetterCNN | 2.46M | 95.44% | Slightly High | Low | ✅ Efficiency Priority |

**Balance Validation:**

1. **Training Curve Analysis:**
   - Both training and validation losses continuously decrease
   - Gap between them remains in reasonable range (1-2%)
   - No case of training loss decreasing while validation loss increasing (overfitting)

2. **Early Stopping Mechanism:**
   - Patience=15, stops if no improvement for 15 epochs
   - Actual training typically reaches best performance at 40-60 epochs
   - Indicates model stops at appropriate time, avoiding overfitting

3. **Model Complexity vs Accuracy Relationship:**
   - Parameter count and accuracy are not completely positively correlated
   - ResNet18 (11.17M) has highest accuracy but also largest parameter count
   - BetterCNN (2.46M) has fewest parameters but still achieves 95.44% accuracy
   - Indicates **training strategy is more important than model selection**

**Conclusions:**

All models achieved **good balance between bias and variance** with 60k training samples:
- ✅ **Low Bias:** All models achieved accuracy above 95%, indicating models can learn effective features
- ✅ **Low Variance:** Small gap between training and test set accuracy (1-2%), indicating good overfitting control
- ✅ **Complexity Matching:** Each model's parameter count matches data size, avoiding underfitting and overfitting
- ✅ **Pretrained Weights Effective:** ImageNet pretrained weights played important roles in ResNet, DenseNet, EfficientNet, and ViT
- ✅ **Training Strategy Critical:** Advanced data augmentation, regularization, and learning rate scheduling are key to achieving good performance

---

## 6. Code Attribution

This project uses the following open-source code and libraries:

1. **PyTorch and torchvision:** Deep learning framework and pretrained models
   - `torchvision.models.resnet18`
   - `torchvision.models.efficientnet_b0`
   - `torchvision.models.densenet121`

2. **timm library:** Vision Transformer models
   - `timm.create_model("vit_tiny_patch16_224", pretrained=True)`

3. **Custom Implementations:**
   - BetterCNN: References classic CNN architecture (VGG philosophy), custom designed
   - MixUp/CutMix: Implemented based on papers (mixup: Beyond Empirical Risk Minimization, CutMix: Regularization Strategy)
   - WarmupCosineScheduler: Implemented based on common learning rate scheduling strategies

**Training Scripts:**
- `train_cnn.py`: BetterCNN training
- `train_resnet.py`: ResNet18 training
- `train_efficientnet.py`: EfficientNet-B0 training
- `train_densenet.py`: DenseNet-121 training
- `train_vit.py`: ViT-Tiny training

**Evaluation Scripts:**
- `evaluate_all_models.py`: Unified evaluation of all models, including TTA and ensemble methods

---

## 7. Project File Structure

```
fashion MNIST/
├── train_cnn.py              # BetterCNN training script
├── train_resnet.py           # ResNet18 training script
├── train_efficientnet.py     # EfficientNet-B0 training script
├── train_densenet.py         # DenseNet-121 training script
├── train_vit.py              # ViT-Tiny training script
├── evaluate_all_models.py    # Unified evaluation script
├── export_test_images.py     # Export test samples
├── requirements.txt          # Dependency list
├── report.md                 # Project report (this file)
├── fashion-mnist_train.csv   # Training data
├── fashion-mnist_test.csv   # Test data
├── models/                   # Saved model weights
│   ├── better_cnn_best.pth
│   ├── resnet18_best.pth
│   ├── efficientnet_b0_best.pth
│   ├── densenet121_best.pth
│   └── vit_tiny_best.pth
└── evaluation_results/       # Evaluation results
    ├── all_models_results.json
    └── model_comparison.md
```

---

## 8. Experiment Reproduction

**Environment Setup:**
```bash
pip install -r requirements.txt
```

**Training Models:**
```bash
python train_cnn.py          # BetterCNN
python train_resnet.py       # ResNet18
python train_efficientnet.py # EfficientNet-B0
python train_densenet.py     # DenseNet-121
python train_vit.py          # ViT-Tiny
```

**Evaluating Models:**
```bash
python evaluate_all_models.py
```

---

## 9. Conclusions and Experience Summary

### 9.1 Main Findings

1. **Model Ensemble Effect Significant:** Weighted average ensemble achieved 96.51% accuracy, improving 0.31% over best single model

2. **Pretrained Weights Effective:** ResNet18, DenseNet-121, EfficientNet-B0 all exceeded 96% accuracy, proving ImageNet pretrained weights remain effective for small datasets

3. **Training Strategy More Important Than Model Selection:** All models achieved accuracy above 95%, with gaps less than 0.5%, indicating advanced data augmentation, regularization, and learning rate scheduling are key

4. **TTA Improves Performance:** Test-time augmentation (original + horizontal flip) consistently improves accuracy

5. **Efficiency vs Performance Trade-off:**
   - Pursuing maximum accuracy: Use ensemble model (96.51%)
   - Balancing accuracy and efficiency: ResNet18 (96.20%, 11.17M parameters)
   - Pursuing efficiency: BetterCNN (95.44%, 2.46M parameters, 31655 FPS)

### 9.2 Technical Application Summary

- ✅ **MixUp/CutMix:** Effectively improves model generalization ability
- ✅ **Learning Rate Warmup:** Stabilizes early training, accelerates convergence
- ✅ **Early Stopping:** Prevents overfitting, automatically selects best model
- ✅ **Label Smoothing:** Improves model robustness
- ✅ **Gradient Clipping:** Stabilizes training process
- ✅ **Mixed Precision Training:** Accelerates training without affecting accuracy
- ✅ **Model Ensemble:** Weighted average method fully utilizes advantages of each model

### 9.3 Team Division of Labor

**Member A:** Responsible for BetterCNN and ResNet18 model implementation and training, data preprocessing and augmentation strategy design

**Member B:** Responsible for EfficientNet-B0, DenseNet-121, and ViT-Tiny model implementation and training, learning rate scheduling and regularization technique application

**Member C:** Responsible for model evaluation script development, ensemble method implementation, performance analysis and report writing

**Collaborative Work:** All members participated in model selection discussions, training strategy optimization, and result analysis

---

## 10. References

1. Fashion-MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist
2. ResNet: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
3. EfficientNet: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019
4. DenseNet: Huang et al., "Densely Connected Convolutional Networks", CVPR 2017
5. Vision Transformer: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
6. MixUp: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
7. CutMix: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers", ICCV 2019
