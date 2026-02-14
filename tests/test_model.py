import os
import sys
import unittest
import torch
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.face_extractor import FaceExtractor
from models.deepfake_detector import DeepfakeDetector
from models.heatmap_generator import HeatmapGenerator

class TestFaceExtractor(unittest.TestCase):
    
    def setUp(self):
        self.extractor = FaceExtractor()
        self.dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_initialization(self):
        self.assertIsNotNone(self.extractor)
        self.assertIsNotNone(self.extractor.face_detector)
    
    def test_extract_faces(self):
        faces = self.extractor.extract_faces(self.dummy_image)
        self.assertIsInstance(faces, list)
    
    def test_preprocess_face(self):
        dummy_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tensor = self.extractor.preprocess_face(dummy_face)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

class TestDeepfakeDetector(unittest.TestCase):
    
    def setUp(self):
        self.detector = DeepfakeDetector()
    
    def test_initialization(self):
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.model)
    
    def test_forward_pass(self):
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = self.detector.forward(dummy_input)
        self.assertEqual(output.shape, (1, 2))

class TestHeatmapGenerator(unittest.TestCase):
    
    def setUp(self):
        self.heatmap_gen = HeatmapGenerator()
        self.dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        self.dummy_tensor = torch.randn(1, 3, 224, 224)
        self.detector = DeepfakeDetector()
    
    def test_initialization(self):
        self.assertIsNotNone(self.heatmap_gen)
    
    def test_fallback_heatmap(self):
        heatmap = self.heatmap_gen._create_fallback_heatmap(self.dummy_image)
        self.assertIsInstance(heatmap, str)

if __name__ == '__main__':
    unittest.main()