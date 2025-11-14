import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

DATA_DIR = "data"
BATCH_SIZE = 16
NUM_EPOCHS = 15  # Increased from 5
NUM_CLASSES = 3  # healthy, minor_damage, severe_damage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Enhanced data augmentation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transforms)
val_dataset = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes

def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Freeze early layers, unfreeze last 2 blocks for fine-tuning
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze layer4 and layer3 for better adaptation
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.layer3.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

def train():
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # Different learning rates for different layers
    optimizer = torch.optim.Adam([
        {'params': model.fc.parameters(), 'lr': 1e-3},
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.layer3.parameters(), 'lr': 1e-4},
    ])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 5

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)
        val_acc = correct / total

        # Learning rate scheduling
        scheduler.step(epoch_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - loss: {epoch_loss:.4f}, val_acc: {val_acc:.4f}")

        # Early stopping and best model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                },
                "models/blade_resnet50.pth",
            )
            print(f"  → New best validation accuracy: {best_val_acc:.4f}, model saved!")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
                break

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    print("Model saved to models/blade_resnet50.pth")

if __name__ == "__main__":
    train()
