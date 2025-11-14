# BladeGuard Architecture

## System Overview

```
┌─────────────┐
│   Client    │ (Drone/Operator)
│  Uploads    │
│   Image     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         FastAPI Server               │
│  ┌──────────────────────────────┐   │
│  │  /analyze-blade endpoint     │   │
│  │  - Receives image + type     │   │
│  └──────────┬───────────────────┘   │
└─────────────┼────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      Image Preprocessing             │
│  - Load from bytes                   │
│  - Resize to 224x224                 │
│  - Normalize (ImageNet stats)        │
│  - Convert to tensor                 │
└─────────────┬────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      ResNet50 Model                  │
│  - Pretrained on ImageNet            │
│  - Fine-tuned on blade images       │
│  - Output: 3-class probabilities     │
└─────────────┬────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      Decision Logic Layer            │
│  - Input: severity + confidence      │
│  - Blade type: TPI or LM            │
│  - Output: operational action        │
│    (no_action/monitor/investigate/  │
│     shutdown_and_investigate)        │
└─────────────┬────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      Grad-CAM Visualization          │
│  - Compute attention map             │
│  - Overlay on original image         │
│  - Save heatmap                      │
└─────────────┬────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      JSON Response                   │
│  - severity, confidence              │
│  - recommended_action                │
│  - needs_shutdown, needs_investigate │
│  - heatmap_path                      │
└─────────────────────────────────────┘
```

## Components

### 1. API Layer (`app/main.py`)
- FastAPI application
- CORS middleware
- File upload handling
- Async endpoints

### 2. Inference Pipeline (`app/inference.py`)
- Image preprocessing
- Model prediction
- Decision logic application
- Heatmap generation

### 3. Model Loader (`app/model_loader.py`)
- Model initialization
- Checkpoint loading
- Device management (CPU/CUDA)
- Graceful fallback

### 4. Preprocessing (`app/utils/preprocessing.py`)
- Image loading from bytes
- Torchvision transforms
- Tensor conversion
- Batch dimension addition

### 5. Visualization (`app/utils/visualization.py`)
- Grad-CAM implementation
- Attention map computation
- Heatmap overlay
- Fallback to dummy visualization

### 6. Decision Logic (`app/inference.py::decide_action`)
- Rule-based system
- Blade-type-specific thresholds
- Confidence-based actions
- Human-readable reasons

## Data Flow

1. **Input**: Image file + blade_type (TPI/LM)
2. **Preprocessing**: PIL Image → Tensor (224x224, normalized)
3. **Inference**: Tensor → Model → Probabilities → Class + Confidence
4. **Decision**: Class + Confidence + Blade Type → Action
5. **Visualization**: Image + Model → Grad-CAM → Heatmap
6. **Output**: JSON with all fields + heatmap path

## Model Architecture

- **Base**: ResNet50 (ImageNet pretrained)
- **Fine-tuning**: Last 2 convolutional blocks (layer3, layer4) + classifier
- **Input**: 224x224 RGB images
- **Output**: 3-class softmax (healthy, minor_damage, severe_damage)
- **Training**: Transfer learning with frozen early layers

## Deployment Considerations

- **Containerization**: Dockerfile provided
- **Scalability**: Stateless API, can be horizontally scaled
- **Monitoring**: Logging and error handling in place
- **Model Versioning**: Model files in `models/` directory
- **Storage**: Results and uploads directories for file management

