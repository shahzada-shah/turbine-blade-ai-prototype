#!/bin/bash
# Quick test script for BladeGuard API

echo "🔍 Testing BladeGuard API..."
echo ""

# Test with severe damage
echo "Testing with severe damage image..."
RESPONSE=$(curl -s -X POST "http://127.0.0.1:8000/analyze-blade" \
  -F "file=@data/test/severe_damage/blade_01.png" \
  -F "blade_type=TPI")

echo "$RESPONSE" | python3 -m json.tool

# Extract heatmap path and open it
HEATMAP=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['heatmap_path'].split('/')[-1])" 2>/dev/null)

if [ ! -z "$HEATMAP" ]; then
    echo ""
    echo "✅ Opening heatmap: http://127.0.0.1:8000/view-heatmap/$HEATMAP"
    open "http://127.0.0.1:8000/view-heatmap/$HEATMAP"
fi

