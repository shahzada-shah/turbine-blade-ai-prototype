# How to Verify Grad-CAM is Working

## Quick Test Script

Run the automated test:
```bash
python3 test_gradcam.py
```

This will:
- ✅ Test if Grad-CAM generates different output than dummy heatmap
- ✅ Compare file sizes and pixel differences
- ✅ Verify color variation (real attention maps have variation)

## Visual Inspection

### Real Grad-CAM Characteristics:
1. **Color variation**: Should show gradients of red/yellow/blue (not just a red circle)
2. **Attention regions**: Hot spots (red/yellow) show where model focuses
3. **Smooth transitions**: Colors blend smoothly across the image
4. **Different per image**: Each image should have unique attention patterns

### Dummy Heatmap Characteristics:
1. **Simple red circle**: Just an outline in the center
2. **No variation**: Same pattern for every image
3. **No attention mapping**: Doesn't reflect model's actual focus

## API Test

Test through the API:
```bash
curl -X POST "http://127.0.0.1:8000/analyze-blade" \
  -F "file=@data/test/severe_damage/blade_01.png" \
  -F "blade_type=TPI"
```

Then check the heatmap file in `results/` directory. It should:
- Show color gradients (red/yellow/blue)
- Highlight regions where damage is detected
- Be different for each image

## File Size Comparison

- **Dummy heatmap**: ~1.5MB (simple PNG)
- **Grad-CAM heatmap**: ~1.3-1.4MB (complex visualization with gradients)

Both are similar size, but Grad-CAM has more visual complexity.

## What Success Looks Like

✅ **Working Grad-CAM:**
- Mean pixel difference > 10 (vs dummy)
- Color std dev > 20 (has variation)
- Visual inspection shows gradients and attention regions

❌ **Not Working:**
- Mean pixel difference ≈ 0
- Same pattern for every image
- Just a red circle outline

## Debugging

If Grad-CAM fails, check:
1. Model is loaded: `model_loader.model is not None`
2. Tensor has gradients: `tensor.requires_grad == True`
3. Hooks are registered: Check for errors in console
4. Fallback message: Look for "Grad-CAM failed (falling back to dummy)" in logs

