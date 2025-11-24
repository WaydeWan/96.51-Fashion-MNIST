# Model Comparison Table

**Note:** All models used TTA (Test-Time Augmentation), ensemble model uses weighted average method.

| Model | Accuracy | Parameters | Model Size | Inference Speed | Correct | TTA |
|-------|----------|------------|------------|-----------------|---------|-----|
| Ensemble (Weighted Average) | 0.9651 | 30.14M | 115.49MB | N/A (Ensemble) | 9651/10000 | N/A |
| ResNet18 | 0.9620 | 11.17M | 42.66MB | 9754.4 FPS | 9620/10000 | Yes |
| DenseNet-121 | 0.9612 | 6.96M | 26.85MB | 2368.8 FPS | 9612/10000 | Yes |
| EfficientNet-B0 | 0.9602 | 4.02M | 15.49MB | 3742.7 FPS | 9602/10000 | Yes |
| ViT-Tiny | 0.9578 | 5.53M | 21.08MB | 1673.9 FPS | 9578/10000 | Yes |
| BetterCNN | 0.9544 | 2.46M | 9.40MB | 29719.7 FPS | 9544/10000 | Yes |
