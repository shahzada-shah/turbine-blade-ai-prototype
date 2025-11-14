#!/usr/bin/env python3
"""
Enhanced model evaluation with confusion matrix visualization
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from app.model_loader import model_loader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

DATA_DIR = "data/test"

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="results/confusion_matrix.png"):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - BladeGuard Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def plot_class_distribution(class_names, counts, save_path="results/class_distribution.png"):
    """Plot class distribution"""
    plt.figure(figsize=(8, 6))
    plt.bar(class_names, counts, color=['green', 'orange', 'red'])
    plt.title('Test Set Class Distribution')
    plt.ylabel('Number of Images')
    plt.xlabel('Class')
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Class distribution saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    if model_loader.model is None:
        print("Error: Model not found. Please train a model first using train_model.py")
        exit(1)

    if not os.path.exists(DATA_DIR):
        print(f"Error: Test directory not found at {DATA_DIR}")
        exit(1)

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_labels = []
    all_preds = []
    all_probs = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Evaluating model on test set...")
    model_loader.model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model_loader.model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    class_names = test_dataset.classes
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"\nClasses: {class_names}")
    print(f"Test set size: {len(all_labels)} images")
    
    # Class distribution
    unique, counts = np.unique(all_labels, return_counts=True)
    print(f"\nClass distribution: {dict(zip([class_names[i] for i in unique], counts))}")
    plot_class_distribution([class_names[i] for i in unique], counts)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    plot_confusion_matrix(all_labels, all_preds, class_names)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(all_preds)[class_mask] == i)
            print(f"  {class_name}: {class_acc:.2%}")
    
    # Overall accuracy
    overall_acc = np.mean(np.array(all_labels) == np.array(all_preds))
    print(f"\nOverall Accuracy: {overall_acc:.2%}")
    
    print("\n" + "="*50)
    print("Evaluation complete! Check results/ for visualizations.")

