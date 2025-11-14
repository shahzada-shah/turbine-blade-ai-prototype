# BladeGuard

AI-powered blade inspection system using computer vision and deep learning for turbine blade defect detection and operational decision support.

## Overview

BladeGuard is a FastAPI-based service that analyzes turbine blade images from drone inspections to detect damage and provide actionable recommendations for ROC operators and site managers. The system combines ML-based severity classification with rule-based decision logic that considers blade type (TPI/LM) and confidence thresholds to recommend operational actions.

## Key Features

- **Blade-Type-Aware Analysis**: Accepts blade type specification (TPI or LM) for manufacturer-specific decision thresholds
- **Operational Decision Logic**: Converts ML predictions into clear recommendations:
  - `no_action` – No visible damage detected
  - `monitor` – Low-confidence minor damage, continue monitoring
  - `investigate` – Requires manual review before action
  - `shutdown_and_investigate` – High-confidence severe damage, immediate action required
- **Structured API Response**: Returns `recommended_action`, `needs_shutdown`, `needs_investigation`, and `action_reason` for UI integration
- **Visualization**: Generates heatmap overlays highlighting detected damage areas
- **Color-Coded Alerts**: Response format enables green/yellow/red UI indicators for operators

## Blade-Type-Aware Decision Logic

The system applies manufacturer-specific thresholds when converting severity predictions into operational recommendations:

| Blade Type | Severity | Confidence | Recommended Action |
|------------|----------|------------|-------------------|
| Any | healthy | any | no_action |
| Any | minor_damage | < 0.8 | monitor |
| Any | minor_damage | ≥ 0.8 | investigate |
| TPI | severe_damage | ≥ 0.75 | shutdown_and_investigate |
| LM | severe_damage | ≥ 0.85 | shutdown_and_investigate |
| Any | severe_damage | < threshold | investigate |

## Project Structure

```
bladeguard/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application with blade-type endpoint
│   ├── inference.py         # Inference logic + decision layer
│   ├── model_loader.py      # Model loading utilities
│   └── utils/
│       ├── preprocessing.py  # Image preprocessing (torchvision-ready)
│       └── visualization.py  # Heatmap generation
├── models/
│   └── placeholder_model.pth # Trained model file (placeholder)
├── uploads/                  # Uploaded images storage
├── results/                  # Analysis results & heatmaps
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the API Server

Start the FastAPI server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - Root endpoint, returns API status
- `POST /analyze-blade` - Analyze blade image with blade type specification
  - **Parameters:**
    - `file`: Image file (multipart/form-data)
    - `blade_type`: "TPI" or "LM" (form field, defaults to "UNKNOWN")
  - **Response:**
    ```json
    {
      "blade_type": "TPI",
      "damage_detected": true,
      "severity": "severe_damage",
      "confidence": 0.93,
      "recommended_action": "shutdown_and_investigate",
      "needs_shutdown": true,
      "needs_investigation": true,
      "action_reason": "High-confidence severe damage on TPI blade – recommend immediate shutdown and inspection.",
      "heatmap_path": "results/heatmap_abc123.png"
    }
    ```
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

### Testing the API

**Using curl:**
```bash
curl -X POST "http://127.0.0.1:8000/analyze-blade" \
  -F "file=@path/to/blade_image.jpg" \
  -F "blade_type=TPI"
```

**Using the interactive docs:**
1. Visit http://127.0.0.1:8000/docs
2. Click on `/analyze-blade` endpoint
3. Click "Try it out"
4. Upload an image and specify blade_type
5. Execute and view the response

## Development

### Current Status

The system currently uses random predictions for testing. The inference pipeline is structured to easily integrate a real ML model:

- Preprocessing utilities are ready for PyTorch/torchvision models
- Decision logic layer is fully implemented and tested
- API interface is production-ready

### Model Integration

To integrate a real ML model:

1. Place your trained PyTorch model in the `models/` directory
2. Update `model_loader.py` to load your specific model architecture
3. Replace the random prediction section in `inference.py` with actual model inference
4. The decision logic will automatically work with real predictions

### Image Processing

The preprocessing pipeline uses torchvision transforms (optional import) and is ready for CNN models:
- Resize to 224x224
- Convert to tensor
- Normalize with ImageNet statistics
- Add batch dimension

## Technical Highlights

- **Rule-Based Decision Layer**: Transparent, auditable decision logic on top of ML predictions
- **Manufacturer-Specific Thresholds**: Different confidence requirements for TPI vs LM blades
- **Structured Output**: API response designed for direct UI consumption with color-coded alerts
- **Production-Ready Architecture**: FastAPI with CORS, async endpoints, and comprehensive error handling

## Resume / Portfolio Bullet

> Designed a blade-type-aware decision layer for the BladeGuard API that maps ML predictions (severity + confidence) into operational actions (no_action, monitor, investigate, shutdown_and_investigate) using TPI/LM-specific thresholds, exposing structured fields (recommended_action, needs_shutdown, needs_investigation, action_reason) for a color-coded UI used by ROC operators and site managers.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
