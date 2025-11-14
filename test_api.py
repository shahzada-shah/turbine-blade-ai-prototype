#!/usr/bin/env python3
"""
Simple test script for BladeGuard API
"""
import requests
import sys
from pathlib import Path

API_URL = "http://127.0.0.1:8000"

def test_root_endpoint():
    """Test the root endpoint"""
    print("Testing root endpoint...")
    try:
        response = requests.get(f"{API_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_analyze_blade(image_path):
    """Test the analyze-blade endpoint with an image"""
    print(f"\nTesting analyze-blade endpoint with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(f"{API_URL}/analyze-blade", files=files)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("BladeGuard API Test Script")
    print("=" * 40)
    
    # Test root endpoint
    if not test_root_endpoint():
        print("\n❌ Root endpoint test failed. Is the server running?")
        print("Start the server with: python3 -m uvicorn app.main:app --reload")
        sys.exit(1)
    
    # Test analyze-blade endpoint if image provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_analyze_blade(image_path)
    else:
        print("\n💡 To test image analysis, provide an image path:")
        print("   python3 test_api.py path/to/image.jpg")

