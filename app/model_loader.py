import torch
from torchvision import models
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join("models", "blade_resnet50.pth")

class ModelLoader:
    def __init__(self):
        # Check if model exists, otherwise use placeholder
        if not os.path.exists(MODEL_PATH):
            self.model = None
            self.class_names = ["healthy", "minor_damage", "severe_damage"]
            return
        
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        class_names = checkpoint["class_names"]

        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(DEVICE)

        self.model = model
        self.class_names = class_names

    def predict(self, tensor):
        # Fallback to placeholder if model not loaded
        if self.model is None:
            return "healthy", 0.95
        
        tensor = tensor.to(DEVICE)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
        label = self.class_names[preds.item()]
        confidence = conf.item()
        return label, confidence

model_loader = ModelLoader()
