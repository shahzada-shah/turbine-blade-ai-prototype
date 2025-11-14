#!/usr/bin/env python3
"""
Test script to verify Grad-CAM is actually working
"""
import os
import sys
from PIL import Image
import numpy as np
from app.model_loader import model_loader
from app.utils.preprocessing import load_image_from_bytes, preprocess_for_model
from app.utils.visualization import create_gradcam_heatmap, create_dummy_heatmap

def test_gradcam():
    """Test if Grad-CAM is working vs dummy heatmap"""
    
    if model_loader.model is None:
        print("❌ ERROR: Model not loaded. Train a model first.")
        return False
    
    # Find a test image
    test_image_path = None
    for root, dirs, files in os.walk("data/test"):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                test_image_path = os.path.join(root, file)
                break
        if test_image_path:
            break
    
    if not test_image_path:
        print("❌ ERROR: No test images found in data/test/")
        return False
    
    print(f"📸 Testing with image: {test_image_path}")
    
    # Load image
    with open(test_image_path, 'rb') as f:
        image_bytes = f.read()
    
    pil_image = load_image_from_bytes(image_bytes)
    tensor = preprocess_for_model(pil_image)
    
    # Get prediction
    severity, confidence = model_loader.predict(tensor)
    print(f"📊 Prediction: {severity} (confidence: {confidence:.2f})")
    
    # Get class index
    target_class_idx = model_loader.class_names.index(severity)
    
    # Test Grad-CAM
    gradcam_path = "results/test_gradcam.png"
    dummy_path = "results/test_dummy.png"
    
    print("\n🔍 Testing Grad-CAM...")
    try:
        create_gradcam_heatmap(
            model_loader.model,
            tensor,
            pil_image,
            gradcam_path,
            target_class_idx=target_class_idx
        )
        
        # Create dummy for comparison
        create_dummy_heatmap(pil_image, dummy_path)
        
        # Compare file sizes (Grad-CAM should be larger/more complex)
        gradcam_size = os.path.getsize(gradcam_path)
        dummy_size = os.path.getsize(dummy_path)
        
        print(f"   Grad-CAM file size: {gradcam_size:,} bytes")
        print(f"   Dummy file size: {dummy_size:,} bytes")
        
        # Load and compare images
        gradcam_img = Image.open(gradcam_path)
        dummy_img = Image.open(dummy_path)
        
        gradcam_array = np.array(gradcam_img)
        dummy_array = np.array(dummy_img)
        
        # Check if they're different (Grad-CAM should have more variation)
        diff = np.abs(gradcam_array.astype(float) - dummy_array.astype(float))
        mean_diff = np.mean(diff)
        
        print(f"   Mean pixel difference: {mean_diff:.2f}")
        
        # Check if Grad-CAM has color variation (not just a red circle)
        gradcam_std = np.std(gradcam_array)
        dummy_std = np.std(dummy_array)
        
        print(f"   Grad-CAM color std dev: {gradcam_std:.2f}")
        print(f"   Dummy color std dev: {dummy_std:.2f}")
        
        # Verification criteria
        is_working = (
            mean_diff > 10 and  # Should be significantly different
            gradcam_std > 20    # Should have color variation
        )
        
        if is_working:
            print("\n✅ SUCCESS: Grad-CAM appears to be working!")
            print("   - Heatmap is different from dummy")
            print("   - Has color variation (real attention map)")
            print(f"   - Check {gradcam_path} to see the visualization")
            return True
        else:
            print("\n⚠️  WARNING: Grad-CAM may not be working correctly")
            print("   - Heatmap is too similar to dummy")
            print("   - May be falling back to dummy heatmap")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: Grad-CAM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Grad-CAM Verification Test")
    print("=" * 50)
    success = test_gradcam()
    sys.exit(0 if success else 1)

