# Project Structure

```
bladeguard/
├── app/                    # Main application code
│   ├── main.py            # FastAPI application
│   ├── inference.py       # ML inference + decision logic
│   ├── model_loader.py    # Model loading
│   └── utils/             # Utilities
│       ├── preprocessing.py
│       └── visualization.py
│
├── tests/                  # All test files
│   ├── test_inference.py  # Unit tests
│   ├── test_model.py      # Model evaluation
│   ├── test_api.py        # API testing
│   └── test_gradcam.py    # Grad-CAM verification
│
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md
│   ├── MODEL_CARD.md
│   ├── TESTING.md
│   └── ...
│
├── models/                 # Trained models
├── data/                   # Dataset (gitignored)
├── results/                # Output files (gitignored)
├── uploads/                # Uploaded images (gitignored)
│
├── train_model.py          # Training script
├── evaluate_model.py       # Enhanced evaluation
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose
├── requirements.txt        # Dependencies
└── README.md              # Main documentation
```

## Running Tests

All test files are in the `tests/` directory. Run from project root:

```bash
# Model evaluation
python tests/test_model.py

# Enhanced evaluation with visualizations
python evaluate_model.py

# API testing
python tests/test_api.py <image_path>

# Grad-CAM verification
python tests/test_gradcam.py

# Unit tests
pytest tests/
```

## Documentation

All documentation is in the `docs/` directory:
- `ARCHITECTURE.md` - System architecture
- `MODEL_CARD.md` - Model documentation
- `TESTING.md` - Testing guide
- `DATA_STRUCTURE.md` - Dataset organization
- `VERIFY_GRADCAM.md` - Grad-CAM verification
- `VIEW_HEATMAP.md` - Heatmap viewing guide

