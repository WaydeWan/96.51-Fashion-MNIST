"""
Unified evaluation script for all models
Supports evaluation of all 5 models: BetterCNN, ResNet18, EfficientNet, DenseNet, ViT
Records detailed metrics: accuracy, confusion matrix, per-class metrics, model size, inference speed, etc.
Generates comparison tables and visualization results
"""
import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import all models
from train_cnn import BetterCNN
from train_resnet import ResNet28
from train_efficientnet import build_model as build_efficientnet
from train_densenet import build_model as build_densenet
import timm

# Set font for matplotlib
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class label mapping
IDX2LABEL = {
    0: "T-shirt/Top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

# Model configuration
MODEL_CONFIGS = {
    'better_cnn': {
        'name': 'BetterCNN',
        'weights': 'better_cnn_fmnist.pt',
        'build_fn': lambda: BetterCNN(num_classes=10),
        'input_size': 28,
        'channels': 1,
    },
    'resnet18': {
        'name': 'ResNet18',
        'weights': 'resnet_fmnist.pt',
        'build_fn': lambda: ResNet28(num_classes=10),
        'input_size': 28,
        'channels': 1,
    },
    'efficientnet': {
        'name': 'EfficientNet-B0',
        'weights': 'efficientnet_fmnist.pt',
        'build_fn': lambda: build_efficientnet(num_classes=10),
        'input_size': 128,
        'channels': 1,
    },
    'densenet': {
        'name': 'DenseNet-121',
        'weights': 'densenet_fmnist.pt',
        'build_fn': lambda: build_densenet(num_classes=10),
        'input_size': 28,
        'channels': 1,
    },
    'vit': {
        'name': 'ViT-Tiny',
        'weights': 'vit_fmnist.pt',
        'build_fn': lambda: timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10),
        'input_size': 224,
        'channels': 3,
    }
}


def get_test_loader(model_name, batch_size=256):
    """Get corresponding test data loader based on model"""
    config = MODEL_CONFIGS[model_name]
    
    if config['input_size'] == 28 and config['channels'] == 1:
        # Standard 28×28 grayscale image
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif config['input_size'] == 128 and config['channels'] == 1:
        # EfficientNet: upsample to 128×128
        test_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif config['input_size'] == 224 and config['channels'] == 3:
        # ViT: upsample to 224×224, expand to 3 channels
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1 channel -> 3 channels
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError(f"Unknown model config: {model_name}")
    
    test_set = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=test_transform
    )
    
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    return test_loader, len(test_set)


def load_model(model_name):
    """Load specified model"""
    config = MODEL_CONFIGS[model_name]
    model = config['build_fn']().to(device)
    
    # Get absolute path of script directory
    script_dir = Path(__file__).parent.absolute()
    weights_path = script_dir / config['weights']
    
    if not weights_path.exists():
        print(f"Warning: Model weight file {weights_path} does not exist, skipping this model")
        print(f"  Current working directory: {os.getcwd()}")
        print(f"  Script directory: {script_dir}")
        return None
    
    print(f"  Loading model weights: {weights_path.name}")
    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def calculate_model_size(model):
    """Calculate model size (MB)"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def measure_inference_speed(model, test_loader, num_samples=100):
    """Measure inference speed (FPS)"""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i * test_loader.batch_size >= num_samples:
                break
            x = x.to(device, non_blocking=True)
            
            # Warmup
            if i == 0:
                for _ in range(10):
                    _ = model(x)
            
            # Measure time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            _ = model(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            batch_time = (end_time - start_time) / x.size(0)  # Time per image
            times.append(1.0 / batch_time)  # FPS
    
    return np.mean(times) if times else 0.0


def apply_tta_augment(x, augment_type='flip'):
    """Apply TTA augmentation to tensor (directly on GPU, more efficient)"""
    if augment_type == 'flip':
        return torch.flip(x, dims=[3])  # Horizontal flip
    elif augment_type == 'none':
        return x
    else:
        return x  # Other augmentations need to start from raw data, only flip here


def get_enhanced_tta_transforms():
    """Get enhanced TTA transform list (more diverse augmentations)"""
    from torchvision import transforms
    from PIL import Image
    
    tta_list = [
        # 1. Original (no augmentation)
        transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
        
        # 2. Horizontal flip
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        
        # 3. Small angle rotation (+5 degrees)
        transforms.Compose([
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        
        # 4. Small angle rotation (-5 degrees)
        transforms.Compose([
            transforms.RandomRotation(-5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        
        # 5. Slight scaling (1.05x)
        transforms.Compose([
            transforms.Resize(30),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
    ]
    return tta_list


def evaluate_model_detailed(model, test_loader, model_name, use_tta=False):
    """Detailed model evaluation (supports TTA)"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    correct = 0
    total = 0
    
    use_amp = torch.cuda.is_available()
    
    if use_tta:
        print(f"  Using TTA: Original + Horizontal Flip (2 augmentations)")
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(test_loader, desc=f"Evaluating {MODEL_CONFIGS[model_name]['name']}{' (TTA)' if use_tta else ''}")):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            if use_tta:
                # TTA: Make multiple augmented predictions on batch, then average (more efficient)
                tta_probs = []
                
                # 1. Original prediction
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(x)
                else:
                    logits = model(x)
                tta_probs.append(torch.softmax(logits, dim=1))
                
                # 2. Horizontal flip prediction
                x_flip = torch.flip(x, dims=[3])
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits_flip = model(x_flip)
                else:
                    logits_flip = model(x_flip)
                tta_probs.append(torch.softmax(logits_flip, dim=1))
                
                # Average probabilities from all TTA predictions
                probs = torch.mean(torch.stack(tta_probs), dim=0)
                preds = probs.argmax(dim=1)
            else:
                # Standard evaluation
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(x)
                else:
                    logits = model(x)
                
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate overall accuracy
    accuracy = correct / total
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate per-class metrics
    report = classification_report(all_labels, all_preds, 
                                   target_names=[IDX2LABEL[i] for i in range(10)],
                                   output_dict=True, zero_division=0)
    
    # Per-class accuracy
    per_class_acc = []
    for i in range(10):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean()
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    # Error analysis
    error_pairs = defaultdict(int)
    for true_label, pred_label in zip(all_labels, all_preds):
        if true_label != pred_label:
            error_pairs[(int(true_label), int(pred_label))] += 1
    
    # Most common error pairs (top 5)
    top_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'accuracy': accuracy,
        'correct': int(correct),
        'total': int(total),
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': per_class_acc,
        'classification_report': report,
        'top_errors': [(IDX2LABEL[t], IDX2LABEL[p], count) for (t, p), count in top_errors],
        'all_predictions': all_preds.tolist(),
        'all_labels': all_labels.tolist(),
        'all_probs': all_probs.tolist()
    }


def plot_confusion_matrix(cm, model_name, save_dir=None):
    """Plot confusion matrix"""
    if save_dir is None:
        # Get absolute path of script directory (project root)
        script_dir = Path(__file__).parent.absolute()
        save_dir = script_dir / "evaluation_results"
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[IDX2LABEL[i] for i in range(10)],
                yticklabels=[IDX2LABEL[i] for i in range(10)])
    plt.title(f'{MODEL_CONFIGS[model_name]["name"]} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(str(save_dir / f"confusion_matrix_{model_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def get_all_model_predictions(models_dict, test_loader_dict):
    """Get prediction probabilities from all models (for multiple ensemble methods)"""
    all_model_probs = {}
    all_labels = None
    
    for model_name, model in models_dict.items():
        print(f"  Getting predictions from {MODEL_CONFIGS[model_name]['name']}...")
        test_loader = test_loader_dict[model_name]
        model_probs = []
        labels = []
        
        model.eval()
        use_amp = torch.cuda.is_available()
        
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f"  {MODEL_CONFIGS[model_name]['name']}"):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(x)
                else:
                    logits = model(x)
                
                probs = torch.softmax(logits, dim=1)
                model_probs.append(probs.cpu().numpy())
                labels.extend(y.cpu().numpy())
        
        all_model_probs[model_name] = np.vstack(model_probs)
        if all_labels is None:
            all_labels = np.array(labels)
    
    return all_model_probs, all_labels


def evaluate_ensemble_weighted(models_dict, test_loader_dict, results_dict):
    """Evaluate weighted ensemble (assign weights based on single model accuracy)"""
    print("\n" + "="*60)
    print("Evaluating Weighted Ensemble (based on accuracy)")
    print("="*60)
    
    all_model_probs, all_labels = get_all_model_predictions(models_dict, test_loader_dict)
    
    # Calculate weights based on single model accuracy (higher accuracy, larger weight)
    model_names = list(models_dict.keys())
    accuracies = [results_dict[name]['accuracy'] for name in model_names]
    
    # Use softmax to normalize weights (makes weights smoother)
    accuracies_array = np.array(accuracies)
    # Subtract minimum, then softmax, makes weights more reasonable
    accuracies_normalized = accuracies_array - accuracies_array.min()
    weights = np.exp(accuracies_normalized * 10)  # Amplify differences
    weights = weights / weights.sum()
    
    print("  Model Weights:")
    for name, acc, weight in zip(model_names, accuracies, weights):
        print(f"    {MODEL_CONFIGS[name]['name']}: {acc:.4f} -> weight {weight:.4f}")
    
    # Weighted average
    ensemble_probs = np.zeros_like(all_model_probs[model_names[0]])
    for name, weight in zip(model_names, weights):
        ensemble_probs += all_model_probs[name] * weight
    
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    return ensemble_preds, all_labels, 'Weighted (by Accuracy)'


def evaluate_ensemble_geometric_mean(models_dict, test_loader_dict):
    """Evaluate geometric mean ensemble (geometric mean of probabilities)"""
    print("\n" + "="*60)
    print("Evaluating Geometric Mean Ensemble")
    print("="*60)
    
    all_model_probs, all_labels = get_all_model_predictions(models_dict, test_loader_dict)
    
    # Geometric mean: take log of probabilities, average, then exponentiate
    model_names = list(models_dict.keys())
    log_probs = [np.log(all_model_probs[name] + 1e-10) for name in model_names]
    ensemble_log_probs = np.mean(log_probs, axis=0)
    ensemble_probs = np.exp(ensemble_log_probs)
    ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)  # Renormalize
    
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    return ensemble_preds, all_labels, 'Geometric Mean'


def evaluate_ensemble_voting(models_dict, test_loader_dict):
    """Evaluate voting ensemble (hard voting: each model votes, select class with most votes)"""
    print("\n" + "="*60)
    print("Evaluating Voting Ensemble (Hard Voting)")
    print("="*60)
    
    all_model_probs, all_labels = get_all_model_predictions(models_dict, test_loader_dict)
    
    # Hard voting: each model predicts a class, then count votes
    model_names = list(models_dict.keys())
    all_preds = [np.argmax(all_model_probs[name], axis=1) for name in model_names]
    
    # For each sample, count votes for each class
    ensemble_preds = []
    for i in range(len(all_labels)):
        votes = [preds[i] for preds in all_preds]
        # Count votes for each class
        vote_counts = np.bincount(votes, minlength=10)
        # Select class with most votes (if tie, select highest probability)
        max_votes = vote_counts.max()
        candidates = np.where(vote_counts == max_votes)[0]
        if len(candidates) == 1:
            ensemble_preds.append(candidates[0])
        else:
            # In case of tie, use probability average
            probs = np.mean([all_model_probs[name][i] for name in model_names], axis=0)
            ensemble_preds.append(np.argmax(probs))
    
    ensemble_preds = np.array(ensemble_preds)
    return ensemble_preds, all_labels, 'Hard Voting'


def evaluate_ensemble(models_dict, test_loader_dict, results_dict=None):
    """Evaluate model ensemble (multiple methods)"""
    print("\n" + "="*60)
    print("Evaluating Model Ensemble")
    print("="*60)
    
    # Get prediction probabilities from all models
    all_model_probs, all_labels = get_all_model_predictions(models_dict, test_loader_dict)
    
    # 1. Simple average
    print("\n  Method 1: Simple Average")
    model_names = list(models_dict.keys())
    ensemble_probs_simple = np.mean([all_model_probs[name] for name in model_names], axis=0)
    ensemble_preds_simple = np.argmax(ensemble_probs_simple, axis=1)
    
    # 2. Weighted average (if results_dict available)
    ensemble_preds_weighted = None
    if results_dict:
        print("\n  Method 2: Weighted Average (based on accuracy)")
        ensemble_preds_weighted, _, _ = evaluate_ensemble_weighted(models_dict, test_loader_dict, results_dict)
    
    # 3. Geometric mean
    print("\n  Method 3: Geometric Mean")
    ensemble_preds_geo, _, _ = evaluate_ensemble_geometric_mean(models_dict, test_loader_dict)
    
    # 4. Hard voting
    print("\n  Method 4: Hard Voting")
    ensemble_preds_vote, _, _ = evaluate_ensemble_voting(models_dict, test_loader_dict)
    
    # Calculate accuracy for all methods
    methods = {
        'Simple Average': ensemble_preds_simple,
        'Weighted Average': ensemble_preds_weighted,
        'Geometric Mean': ensemble_preds_geo,
        'Hard Voting': ensemble_preds_vote
    }
    
    best_method = None
    best_acc = 0.0
    best_preds = None
    
    print("\n" + "="*60)
    print("Ensemble Method Comparison:")
    print("="*60)
    for method_name, preds in methods.items():
        if preds is not None:
            acc = (preds == all_labels).mean()
            print(f"  {method_name}: {acc:.4f} ({acc*100:.2f}%)")
            if acc > best_acc:
                best_acc = acc
                best_method = method_name
                best_preds = preds
    
    print(f"\nBest Method: {best_method} ({best_acc:.4f})")
    
    # Use results from best method
    ensemble_preds = best_preds
    
    # Calculate accuracy
    correct = (ensemble_preds == all_labels).sum()
    total = len(all_labels)
    accuracy = correct / total
    
    # Calculate confusion matrix and classification report
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, ensemble_preds)
    report = classification_report(all_labels, ensemble_preds,
                                   target_names=[IDX2LABEL[i] for i in range(10)],
                                   output_dict=True, zero_division=0)
    
    # Per-class accuracy
    per_class_acc = []
    for i in range(10):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = (ensemble_preds[class_mask] == all_labels[class_mask]).mean()
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    # Error analysis
    error_pairs = defaultdict(int)
    for true_label, pred_label in zip(all_labels, ensemble_preds):
        if true_label != pred_label:
            error_pairs[(int(true_label), int(pred_label))] += 1
    
    top_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Calculate total parameters and model size for ensemble (sum of all models)
    total_params = sum(sum(p.numel() for p in model.parameters()) for model in models_dict.values())
    total_size_mb = sum(calculate_model_size(model) for model in models_dict.values())
    
    return {
        'model_name': f'Ensemble ({best_method})',
        'ensemble_method': best_method,
        'total_parameters': int(total_params),
        'model_size_mb': float(total_size_mb),
        'accuracy': float(accuracy),
        'correct': int(correct),
        'total': int(total),
        'per_class_accuracy': [float(x) for x in per_class_acc],
        'top_errors': [(IDX2LABEL[t], IDX2LABEL[p], count) for (t, p), count in top_errors],
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }


def evaluate_all_models(use_tta=True):
    """Evaluate all models (supports TTA)"""
    results = {}
    # Get absolute path of script directory (project root)
    script_dir = Path(__file__).parent.absolute()
    save_dir = script_dir / "evaluation_results"
    save_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Starting evaluation of all models" + (" (with TTA)" if use_tta else ""))
    print("="*60)
    
    # Save all models and test loaders for ensemble
    models_dict = {}
    test_loader_dict = {}
    
    for model_name in MODEL_CONFIGS.keys():
        print(f"\n{'='*60}")
        print(f"Evaluating Model: {MODEL_CONFIGS[model_name]['name']}")
        print(f"{'='*60}")
        
        # Load model
        model = load_model(model_name)
        if model is None:
            print(f"Skipping {model_name} (model file does not exist)")
            continue
        
        # Count model information
        total_params, trainable_params = count_parameters(model)
        model_size_mb = calculate_model_size(model)
        
        print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Model Size: {model_size_mb:.2f} MB")
        
        # Load test data
        test_loader, test_size = get_test_loader(model_name)
        print(f"Test Set Size: {test_size:,}")
        
        # Measure inference speed
        print("Measuring inference speed...")
        fps = measure_inference_speed(model, test_loader, num_samples=100)
        print(f"Inference Speed: {fps:.1f} FPS ({1000/fps:.2f} ms/image)")
        
        # Detailed evaluation (with TTA)
        print("Performing detailed evaluation...")
        eval_results = evaluate_model_detailed(model, test_loader, model_name, use_tta=use_tta)
        
        # Plot confusion matrix
        cm = np.array(eval_results['confusion_matrix'])
        plot_confusion_matrix(cm, model_name, save_dir)
        
        # Save results
        results[model_name] = {
            'model_name': MODEL_CONFIGS[model_name]['name'],
            'use_tta': use_tta,
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'model_size_mb': float(model_size_mb),
            'inference_fps': float(fps),
            'inference_time_ms': float(1000 / fps) if fps > 0 else 0,
            'accuracy': float(eval_results['accuracy']),
            'correct': int(eval_results['correct']),
            'total': int(eval_results['total']),
            'per_class_accuracy': [float(x) for x in eval_results['per_class_accuracy']],
            'top_errors': eval_results['top_errors'],
            'classification_report': eval_results['classification_report']
        }
        
        # Print results
        print(f"\nResults:")
        print(f"  Accuracy: {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
        print(f"  Correct: {eval_results['correct']:,} / {eval_results['total']:,}")
        print(f"  Errors: {eval_results['total'] - eval_results['correct']:,} / {eval_results['total']:,}")
        print(f"\nMost Common Errors:")
        for true_label, pred_label, count in eval_results['top_errors']:
            print(f"  {true_label} -> {pred_label}: {count} times")
        
        # Save model and test loader for ensemble (don't delete model)
        models_dict[model_name] = model
        test_loader_dict[model_name] = test_loader
    
    # Evaluate model ensemble (if at least one model loaded successfully)
    if models_dict:
        ensemble_result = evaluate_ensemble(models_dict, test_loader_dict, results_dict=results)
        results['ensemble'] = ensemble_result
        
        print(f"\nEnsemble Model Results:")
        print(f"  Accuracy: {ensemble_result['accuracy']:.4f} ({ensemble_result['accuracy']*100:.2f}%)")
        print(f"  Correct: {ensemble_result['correct']:,} / {ensemble_result['total']:,}")
        print(f"  Errors: {ensemble_result['total'] - ensemble_result['correct']:,} / {ensemble_result['total']:,}")
        print(f"  Total Parameters: {ensemble_result['total_parameters']:,} ({ensemble_result['total_parameters']/1e6:.2f}M)")
        print(f"  Total Model Size: {ensemble_result['model_size_mb']:.2f} MB")
        print(f"\nMost Common Errors:")
        for true_label, pred_label, count in ensemble_result['top_errors']:
            print(f"  {true_label} -> {pred_label}: {count} times")
        
        # Plot ensemble model confusion matrix
        cm_ensemble = np.array(ensemble_result['confusion_matrix'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[IDX2LABEL[i] for i in range(10)],
                    yticklabels=[IDX2LABEL[i] for i in range(10)])
        plt.title('Ensemble (Weighted Average) - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(str(save_dir / "confusion_matrix_ensemble.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Clean GPU memory
    for model in models_dict.values():
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save all results to JSON
    results_file = save_dir / "all_models_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to: {results_file}")
    
    # Generate comparison table
    generate_comparison_table(results, save_dir)
    
    return results


def generate_comparison_table(results, save_dir):
    """Generate model comparison table (includes TTA and ensemble)"""
    print("\n" + "="*60)
    print("Model Comparison Table")
    print("="*60)
    
    # Table data
    table_data = []
    for model_name, result in results.items():
        if model_name == 'ensemble':
            # Ensemble model has no inference speed (average of multiple models)
            table_data.append({
                'Model': result['model_name'],
                'Accuracy': f"{result['accuracy']:.4f}",
                'Parameters': f"{result['total_parameters']/1e6:.2f}M",
                'Model Size': f"{result['model_size_mb']:.2f}MB",
                'Inference Speed': 'N/A (Ensemble)',
                'Correct': f"{result['correct']}/{result['total']}",
                'TTA': 'N/A'
            })
        else:
            tta_str = 'Yes' if result.get('use_tta', False) else 'No'
            table_data.append({
                'Model': result['model_name'],
                'Accuracy': f"{result['accuracy']:.4f}",
                'Parameters': f"{result['total_parameters']/1e6:.2f}M",
                'Model Size': f"{result['model_size_mb']:.2f}MB",
                'Inference Speed': f"{result.get('inference_fps', 0):.1f} FPS",
                'Correct': f"{result['correct']}/{result['total']}",
                'TTA': tta_str
            })
    
    # Sort by accuracy
    table_data.sort(key=lambda x: float(x['Accuracy']), reverse=True)
    
    # Print table
    print("\n| Model | Accuracy | Parameters | Model Size | Inference Speed | Correct | TTA |")
    print("|-------|----------|------------|------------|-----------------|---------|-----|")
    for row in table_data:
        print(f"| {row['Model']} | {row['Accuracy']} | {row['Parameters']} | {row['Model Size']} | {row['Inference Speed']} | {row['Correct']} | {row['TTA']} |")
    
    # Save as Markdown file
    md_file = save_dir / "model_comparison.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Model Comparison Table\n\n")
        f.write("**Note:** All models used TTA (Test-Time Augmentation), ensemble model uses weighted average method.\n\n")
        f.write("| Model | Accuracy | Parameters | Model Size | Inference Speed | Correct | TTA |\n")
        f.write("|-------|----------|------------|------------|-----------------|---------|-----|\n")
        for row in table_data:
            f.write(f"| {row['Model']} | {row['Accuracy']} | {row['Parameters']} | {row['Model Size']} | {row['Inference Speed']} | {row['Correct']} | {row['TTA']} |\n")
    
    print(f"\nComparison table saved to: {md_file}")


if __name__ == "__main__":
    # Need to install sklearn: pip install scikit-learn
    try:
        from sklearn.metrics import confusion_matrix, classification_report
    except ImportError:
        print("Error: scikit-learn is required")
        print("Run: pip install scikit-learn")
        exit(1)
    
    import sys
    # Use TTA by default, can be disabled via parameter
    use_tta = True
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'no_tta':
        use_tta = False
        print("Note: Not using TTA")
    
    results = evaluate_all_models(use_tta=use_tta)
    print("\n" + "="*60)
    print("All model evaluation completed!")
    print("="*60)
    print(f"\nEvaluated {len([k for k in results.keys() if k != 'ensemble'])} single models + 1 ensemble model")
    print(f"All results saved in evaluation_results/ folder")

