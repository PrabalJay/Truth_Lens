import os
import sys
from models.deepfake_detector import DeepfakeDetector

model_path = "models/saved/ensemble_model.pth"

if not os.path.exists(model_path):
    print(f"❌ Model not found at {model_path}")
    print("Please place ensemble_model.pth in models/saved/")
    sys.exit(1)

print("="*60)
print("🔍 Testing Trained Deepfake Detector")
print("="*60)

detector = DeepfakeDetector(model_path=model_path)

test_images = [
    "data/sample_images/real/real_1.jpg",
    "data/sample_images/fake/fake_1.jpg"
]

for img_path in test_images:
    if os.path.exists(img_path):
        print(f"\n📸 Testing: {img_path}")
        result = detector.analyze_image(img_path)
        print(f"   Result: {result['result']}")
        print(f"   Confidence: {result['confidence']}%")
        print(f"   Faces: {result['faces_analyzed']}")
        print(f"   Message: {result['message']}")
    else:
        print(f"\n⚠️ Test image not found: {img_path}")

print("\n" + "="*60)
print("✅ Phase 1 Complete! Model is ready for Phase 2.")
print("="*60)