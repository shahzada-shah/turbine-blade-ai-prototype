# BladeGuard

AI-powered blade inspection system using computer vision and deep learning.

## Project Structure

```
bladeguard/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── inference.py         # Model inference logic
│   ├── model_loader.py      # Model loading utilities
│   └── utils/
│       ├── preprocessing.py  # Image preprocessing utilities
│       └── visualization.py  # Visualization utilities
├── models/
│   └── placeholder_model.pth # Trained model file
├── uploads/                  # Uploaded images storage
├── results/                  # Analysis results storage
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
cd bladeguard
python -m app.main
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - Root endpoint, returns API status
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## Development

### Adding New Endpoints

Add new routes in `app/main.py` or create separate router modules.

### Model Integration

1. Place your trained PyTorch model in the `models/` directory
2. Update `model_loader.py` to load your specific model architecture
3. Implement inference logic in `inference.py`

### Image Processing

Customize preprocessing steps in `app/utils/preprocessing.py` to match your model's requirements.

## Features (Planned)

- [ ] Image upload and preprocessing
- [ ] Blade defect detection
- [ ] Damage classification
- [ ] Visualization of results
- [ ] Report generation
- [ ] Batch processing

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

