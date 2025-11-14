# Model Card: BladeGuard ResNet50 Classifier

## Model Details
- **Model Name**: BladeGuard ResNet50
- **Version**: 1.0
- **Architecture**: ResNet50 (ImageNet pretrained, fine-tuned)
- **Task**: Multi-class classification (healthy, minor_damage, severe_damage)
- **Framework**: PyTorch 2.8.0

## Training Data
- **Dataset Size**: 42 training images (14 per class), 9 validation images (3 per class), 9 test images (3 per class)
- **Image Resolution**: 224x224
- **Augmentation**: Random horizontal flip, rotation (±15°), color jitter, affine transforms
- **Class Distribution**: Balanced (14:14:14 train, 3:3:3 val/test)

## Performance
- **Overall Accuracy**: 67%
- **Per-Class Metrics** (on test set):
  - Healthy: Precision 0.67, Recall 0.67, F1 0.67
  - Minor Damage: Precision 0.60, Recall 1.00, F1 0.75
  - Severe Damage: Precision 1.00, Recall 0.33, F1 0.50

## Limitations
- **Small Dataset**: Limited to 42 training images - model would benefit from more data
- **Class Imbalance Sensitivity**: Severe damage has lower recall (33%) - may miss some severe cases
- **Generalization**: Trained on specific blade types - may not generalize to all blade manufacturers
- **Confidence Calibration**: Confidence scores may not be perfectly calibrated

## Intended Use
- **Primary Use**: Assist ROC operators in identifying blade damage from drone imagery
- **Not Intended For**: 
  - Final decision-making without human review
  - Real-time safety-critical systems without additional validation
  - Blades from manufacturers not in training data

## Ethical Considerations
- Model predictions are advisory only - final decisions require human oversight
- False negatives (missing severe damage) could have safety implications
- Decision logic includes manufacturer-specific thresholds to account for different safety requirements

## Model Architecture
- **Base**: ResNet50 pretrained on ImageNet
- **Fine-tuning**: Last 2 blocks (layer3, layer4) + classifier head
- **Input**: 224x224 RGB images, ImageNet normalization
- **Output**: 3-class softmax probabilities

## Evaluation Methodology
- Held-out test set (9 images, 3 per class)
- Classification metrics: Precision, Recall, F1-score
- Visual validation via Grad-CAM attention maps

