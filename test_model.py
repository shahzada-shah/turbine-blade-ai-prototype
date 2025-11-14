import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from app.model_loader import model_loader
from sklearn.metrics import classification_report
import os

DATA_DIR = "data/test"

if model_loader.model is None:
    print("Error: Model not found. Please train a model first using train_model.py")
    exit(1)

if not os.path.exists(DATA_DIR):
    print(f"Error: Test directory not found at {DATA_DIR}")
    print("Expected structure: data/test/{healthy,minor_damage,severe_damage}/")
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

device = "cuda" if torch.cuda.is_available() else "cpu"

for inputs, labels in test_loader:
    inputs = inputs.to(device)
    outputs = model_loader.model(inputs)
    _, preds = torch.max(outputs, 1)
    all_labels.extend(labels.numpy())
    all_preds.extend(preds.cpu().numpy())

print("Classes:", test_dataset.classes)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

