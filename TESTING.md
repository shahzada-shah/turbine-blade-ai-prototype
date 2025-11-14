# Testing Guide for BladeGuard

## 1. Automated Grad-CAM Test

**Quick verification that Grad-CAM is working:**
```bash
python3 test_gradcam.py
```

**Expected output:**
```
✅ SUCCESS: Grad-CAM appears to be working!
   - Heatmap is different from dummy
   - Has color variation (real attention map)
```

## 2. Test via API (Command Line)

### Test with different image types:

**Severe damage:**
```bash
curl -X POST "http://127.0.0.1:8000/analyze-blade" \
  -F "file=@data/test/severe_damage/blade_01.png" \
  -F "blade_type=TPI" | python3 -m json.tool
```

**Healthy blade:**
```bash
curl -X POST "http://127.0.0.1:8000/analyze-blade" \
  -F "file=@data/test/healthy/blade_01.png" \
  -F "blade_type=LM" | python3 -m json.tool
```

**Minor damage:**
```bash
curl -X POST "http://127.0.0.1:8000/analyze-blade" \
  -F "file=@data/test/minor_damage/blade_01.png" \
  -F "blade_type=TPI" | python3 -m json.tool
```

## 3. Test via Interactive API Docs

1. Start the server:
   ```bash
   uvicorn app.main:app --reload
   ```

2. Open browser: http://127.0.0.1:8000/docs

3. Click on `/analyze-blade` endpoint

4. Click "Try it out"

5. Upload an image and set `blade_type` to "TPI" or "LM"

6. Click "Execute"

7. Check the response for:
   - `severity`: Should match the image type
   - `confidence`: Should be between 0 and 1
   - `recommended_action`: Should follow decision logic
   - `heatmap_path`: Path to generated heatmap

## 4. Visual Inspection of Heatmaps

**Check the heatmap files in `results/` directory:**

```bash
# List recent heatmaps
ls -lth results/heatmap*.png | head -5

# Open a heatmap to inspect
open results/heatmap_<filename>.png  # macOS
# or
xdg-open results/heatmap_<filename>.png  # Linux
```

**What to look for:**
- ✅ **Real Grad-CAM**: Color gradients (red/yellow/blue), shows attention regions
- ❌ **Dummy**: Just a red circle outline, same for every image

## 5. Test Model Evaluation

**Check model performance:**
```bash
python3 test_model.py
```

**Expected output:**
```
Classification Report:
               precision    recall  f1-score   support
      healthy       0.67      0.67      0.67         3
 minor_damage       0.60      1.00      0.75         3
severe_damage       1.00      0.33      0.50         3
```

## 6. Test Decision Logic

**Verify blade-type-specific thresholds:**

**TPI with high-confidence severe:**
```bash
# Should return shutdown_and_investigate if confidence >= 0.75
curl -X POST "http://127.0.0.1:8000/analyze-blade" \
  -F "file=@data/test/severe_damage/blade_01.png" \
  -F "blade_type=TPI" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Action: {d['recommended_action']}, Shutdown: {d['needs_shutdown']}, Confidence: {d['confidence']}\")"
```

**LM with high-confidence severe:**
```bash
# Should return shutdown_and_investigate if confidence >= 0.85
curl -X POST "http://127.0.0.1:8000/analyze-blade" \
  -F "file=@data/test/severe_damage/blade_01.png" \
  -F "blade_type=LM" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Action: {d['recommended_action']}, Shutdown: {d['needs_shutdown']}, Confidence: {d['confidence']}\")"
```

## 7. End-to-End Test Script

**Test the full pipeline:**
```bash
python3 test_api.py data/test/severe_damage/blade_01.png
```

## Expected Results

### ✅ Working System:
- Grad-CAM test passes
- API returns valid JSON with all fields
- Heatmaps show color gradients (not just red circles)
- Decision logic applies correct thresholds
- Model predictions match image types

### ❌ Issues to Check:
- If Grad-CAM test fails → Check model is loaded
- If API returns errors → Check server logs
- If heatmaps look identical → Grad-CAM may be falling back to dummy
- If predictions are wrong → Retrain model with more data

## Quick Test Checklist

- [ ] Run `python3 test_gradcam.py` → Should pass
- [ ] Test API with severe damage image → Should detect damage
- [ ] Test API with healthy image → Should return no_action
- [ ] Check heatmap files → Should show gradients, not just circles
- [ ] Test TPI vs LM thresholds → Should apply different shutdown rules
- [ ] Verify model evaluation → Should show reasonable accuracy

