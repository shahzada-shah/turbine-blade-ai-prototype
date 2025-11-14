"""
Unit tests for inference module
"""
import pytest
import uuid
from PIL import Image
import io
import numpy as np
from app.inference import analyze_blade_image, decide_action

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (224, 224), color='gray')
    return img

def test_decide_action_healthy():
    """Test decision logic for healthy blades"""
    result = decide_action("TPI", "healthy", 0.95)
    assert result["recommended_action"] == "no_action"
    assert result["needs_shutdown"] == False
    assert result["needs_investigation"] == False

def test_decide_action_severe_tpi_high_confidence():
    """Test shutdown logic for TPI with high confidence"""
    result = decide_action("TPI", "severe_damage", 0.80)
    assert result["recommended_action"] == "shutdown_and_investigate"
    assert result["needs_shutdown"] == True
    assert result["needs_investigation"] == True

def test_decide_action_severe_lm_high_confidence():
    """Test shutdown logic for LM with high confidence"""
    result = decide_action("LM", "severe_damage", 0.90)
    assert result["recommended_action"] == "shutdown_and_investigate"
    assert result["needs_shutdown"] == True

def test_decide_action_severe_below_threshold():
    """Test investigate logic when below shutdown threshold"""
    result = decide_action("TPI", "severe_damage", 0.70)
    assert result["recommended_action"] == "investigate"
    assert result["needs_shutdown"] == False
    assert result["needs_investigation"] == True

def test_decide_action_minor_high_confidence():
    """Test investigate logic for minor damage with high confidence"""
    result = decide_action("TPI", "minor_damage", 0.85)
    assert result["recommended_action"] == "investigate"
    assert result["needs_investigation"] == True

def test_decide_action_minor_low_confidence():
    """Test monitor logic for minor damage with low confidence"""
    result = decide_action("TPI", "minor_damage", 0.70)
    assert result["recommended_action"] == "monitor"
    assert result["needs_investigation"] == False

def test_analyze_blade_image_structure():
    """Test that analyze_blade_image returns correct structure"""
    img = create_test_image()
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    result = analyze_blade_image(img_bytes.getvalue(), "TPI")
    
    # Check required fields
    assert "blade_type" in result
    assert "damage_detected" in result
    assert "severity" in result
    assert "confidence" in result
    assert "heatmap_path" in result
    assert "recommended_action" in result
    assert "needs_shutdown" in result
    assert "needs_investigation" in result
    assert "action_reason" in result
    
    # Check types
    assert isinstance(result["confidence"], (int, float))
    assert isinstance(result["damage_detected"], bool)
    assert result["severity"] in ["healthy", "minor_damage", "severe_damage"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

