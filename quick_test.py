import os
import sys
import torch

print("=" * 60)
print("🔬 TRUTH LENS - Quick Test")
print("=" * 60)

print("\n1. Testing PyTorch...")
try:
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print("   ✅ PyTorch is working!")
except Exception as e:
    print(f"   ❌ PyTorch error: {e}")

print("\n2. Testing OpenCV...")
try:
    import cv2
    print(f"   OpenCV version: {cv2.__version__}")
    print("   ✅ OpenCV is working!")
except Exception as e:
    print(f"   ❌ OpenCV error: {e}")

print("\n3. Testing Face Extractor...")
try:
    from models.face_extractor import FaceExtractor
    extractor = FaceExtractor()
    print("   ✅ FaceExtractor is working!")
except Exception as e:
    print(f"   ❌ FaceExtractor error: {e}")

print("\n4. Testing Deepfake Detector...")
try:
    from models.deepfake_detector import DeepfakeDetector
    detector = DeepfakeDetector()
    print("   ✅ DeepfakeDetector is working!")
except Exception as e:
    print(f"   ❌ DeepfakeDetector error: {e}")

print("\n5. Testing Heatmap Generator...")
try:
    from models.heatmap_generator import HeatmapGenerator
    heatmap_gen = HeatmapGenerator()
    print("   ✅ HeatmapGenerator is working!")
except Exception as e:
    print(f"   ❌ HeatmapGenerator error: {e}")

print("\n" + "=" * 60)
print("✅ Quick test complete!")
print("=" * 60)