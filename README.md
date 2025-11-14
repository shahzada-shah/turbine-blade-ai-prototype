# BladeGuard

AI-powered turbine blade inspection system that detects damage from drone imagery and provides operational recommendations for ROC operators.

## Overview

Production-ready FastAPI service that combines fine-tuned ResNet50 classification with rule-based decision logic. Analyzes blade images to classify damage severity (healthy/minor/severe) and converts predictions into actionable recommendations based on blade type (TPI/LM) and manufacturer-specific confidence thresholds.

## Key Features

- **Real ML Model**: Fine-tuned ResNet50 on blade image dataset with evaluation metrics
- **Explainable AI**: Grad-CAM visualization showing model attention regions
- **Blade-Type-Aware Logic**: Manufacturer-specific thresholds (TPI: 75%, LM: 85%) for shutdown decisions
- **Operational Actions**: Maps predictions to `no_action`, `monitor`, `investigate`, or `shutdown_and_investigate`
- **Production API**: FastAPI with structured JSON responses, error handling, and CORS

## Decision Logic

| Blade Type | Severity | Confidence | Action |
|------------|----------|------------|--------|
| Any | healthy | any | no_action |
| Any | minor_damage | < 0.8 | monitor |
| Any | minor_damage | ≥ 0.8 | investigate |
| TPI | severe_damage | ≥ 0.75 | shutdown_and_investigate |
| LM | severe_damage | ≥ 0.85 | shutdown_and_investigate |

## Technical Stack

- **ML**: PyTorch, ResNet50, torchvision
- **API**: FastAPI, async endpoints
- **Visualization**: Grad-CAM, OpenCV
- **Evaluation**: scikit-learn metrics (precision/recall/F1)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (requires data in data/train/, data/val/, data/test/)
python train_model.py

# Evaluate model
python tests/test_model.py

# Start API
uvicorn app.main:app --reload
```

## Documentation

- **Architecture**: See `docs/ARCHITECTURE.md`
- **Model Card**: See `docs/MODEL_CARD.md`
- **Testing Guide**: See `docs/TESTING.md`
- **Docker Deployment**: See `Dockerfile` and `docker-compose.yml`

## API Example

```bash
curl -X POST "http://127.0.0.1:8000/analyze-blade" \
  -F "file=@blade_image.jpg" \
  -F "blade_type=TPI"
```

**Response:**
```json
{
  "severity": "severe_damage",
  "confidence": 0.93,
  "recommended_action": "shutdown_and_investigate",
  "needs_shutdown": true,
  "heatmap_path": "results/heatmap_abc123.png"
}
```

## Technical Highlights

- **End-to-End ML Pipeline**: Data → Training → Evaluation → Deployment
- **Explainable AI**: Grad-CAM heatmaps showing where model detects damage
- **Business Logic Layer**: Transparent rule-based system converting ML outputs to operational decisions
- **Production Architecture**: Error handling, graceful fallbacks, structured responses

## Resume / Portfolio Bullet

> Designed a blade-type-aware decision layer for the BladeGuard API that maps ML predictions (severity + confidence) into operational actions (no_action, monitor, investigate, shutdown_and_investigate) using TPI/LM-specific thresholds, exposing structured fields (recommended_action, needs_shutdown, needs_investigation, action_reason) for a color-coded UI used by ROC operators and site managers.

## License

MIT License
