# How to View the Heatmap

## Method 1: Using the API (Easiest)

### Step 1: Test the API
```bash
curl -X POST "http://127.0.0.1:8000/analyze-blade" \
  -F "file=@data/test/severe_damage/blade_01.png" \
  -F "blade_type=TPI" | python3 -m json.tool
```

### Step 2: Copy the heatmap_path from the response
Look for: `"heatmap_path": "results/heatmap_xxxxx.png"`

### Step 3: Open the heatmap
```bash
# macOS
open results/heatmap_xxxxx.png

# Linux
xdg-open results/heatmap_xxxxx.png

# Windows
start results/heatmap_xxxxx.png
```

## Method 2: Interactive Browser (Best for Testing)

### Step 1: Open API Docs
Go to: http://127.0.0.1:8000/docs

### Step 2: Use the `/analyze-blade` endpoint
1. Click on `/analyze-blade`
2. Click "Try it out"
3. Click "Choose File" and select an image from `data/test/`
4. Enter `blade_type`: "TPI" or "LM"
5. Click "Execute"

### Step 3: Get the heatmap path
- Copy the `heatmap_path` value from the response
- Open it in your file browser or image viewer

## Method 3: List and Open Recent Heatmaps

```bash
# List the 5 most recent heatmaps
ls -lth results/heatmap*.png | head -5

# Open the most recent one (macOS)
open $(ls -t results/heatmap*.png | head -1)

# Open the most recent one (Linux)
xdg-open $(ls -t results/heatmap*.png | head -1)
```

## Method 4: Quick Test Script

Create a simple script to test and view:

```bash
# Save this as test_and_view.sh
#!/bin/bash
RESPONSE=$(curl -s -X POST "http://127.0.0.1:8000/analyze-blade" \
  -F "file=@$1" \
  -F "blade_type=$2")
HEATMAP=$(echo $RESPONSE | python3 -c "import sys,json; print(json.load(sys.stdin)['heatmap_path'])")
echo "Heatmap saved to: $HEATMAP"
open "$HEATMAP"  # macOS - change to xdg-open for Linux
```

Usage:
```bash
chmod +x test_and_view.sh
./test_and_view.sh data/test/severe_damage/blade_01.png TPI
```

## What to Look For in the Heatmap

### ✅ Real Grad-CAM (Working):
- **Color gradients**: Red/yellow/blue transitions
- **Attention regions**: Hot spots (red/yellow) show where model focuses
- **Smooth blending**: Colors blend naturally with original image
- **Unique patterns**: Each image has different attention regions

### ❌ Dummy Heatmap (Not Working):
- **Simple red circle**: Just an outline in the center
- **No variation**: Same pattern for every image
- **No gradients**: Just a single color outline

## Quick Visual Test

Compare two heatmaps:
```bash
# Generate heatmap for severe damage
curl -X POST "http://127.0.0.1:8000/analyze-blade" \
  -F "file=@data/test/severe_damage/blade_01.png" \
  -F "blade_type=TPI" > /dev/null

# Generate heatmap for healthy
curl -X POST "http://127.0.0.1:8000/analyze-blade" \
  -F "file=@data/test/healthy/blade_01.png" \
  -F "blade_type=LM" > /dev/null

# Open both to compare
open results/heatmap*.png
```

If they look different (different attention patterns), Grad-CAM is working!

## Troubleshooting

**Can't find the heatmap file?**
- Check the `results/` directory exists
- Verify the path in the API response
- Make sure the server is running

**Heatmap looks like just a red circle?**
- Run `python3 test_gradcam.py` to verify Grad-CAM is working
- Check server logs for errors
- Make sure model is loaded (`models/blade_resnet50.pth` exists)

**Want to see multiple heatmaps?**
```bash
# Open all recent heatmaps
open results/heatmap*.png  # macOS
```

