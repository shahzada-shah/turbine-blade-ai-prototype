# Tests

All test files should be run from the project root directory.

## Running Tests

```bash
# From project root
cd /path/to/bladeguard

# Model evaluation
python tests/test_model.py

# Enhanced evaluation with visualizations
python evaluate_model.py

# API testing (requires server running)
python tests/test_api.py <image_path>

# Grad-CAM verification
PYTHONPATH=. python tests/test_gradcam.py

# Unit tests
pytest tests/
```

## Test Files

- `test_inference.py` - Unit tests for decision logic and API structure
- `test_model.py` - Model evaluation on test set
- `test_api.py` - End-to-end API testing
- `test_gradcam.py` - Grad-CAM visualization verification

