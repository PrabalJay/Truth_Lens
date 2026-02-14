import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np
import logging
from .face_extractor import FaceExtractor
from .heatmap_generator import HeatmapGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch, channels, height, width = x.size()
        
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        
        value = self.value(x).view(batch, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        
        return self.gamma * out + x

class DeepfakeDetector:
    
    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"Initializing DeepfakeDetector on {self.device}")
        
        self.face_extractor = FaceExtractor(self.device)
        self.heatmap_generator = HeatmapGenerator(self.device)
        
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()
        
        if model_path:
            self.load_model(model_path)
            
    def _build_model(self):
        
        try:
            from torchvision.models import EfficientNet_B3_Weights
            weights = EfficientNet_B3_Weights.DEFAULT
            self.efficientnet = models.efficientnet_b3(weights=weights)
            logger.info("✅ Loaded pretrained EfficientNet-B3 with DEFAULT weights")
        except:
            try:
                self.efficientnet = models.efficientnet_b3(pretrained=True)
                logger.info("✅ Loaded pretrained EfficientNet-B3 with pretrained=True")
            except Exception as e:
                self.efficientnet = models.efficientnet_b3(pretrained=False)
                logger.warning(f"Using random weights: {e}")

        self.efficientnet.classifier = nn.Identity()
        self.attention_eff = SpatialAttention(1536)
        
        self.classifier = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
        
        model = nn.Module()
        model.efficientnet = self.efficientnet
        model.attention_eff = self.attention_eff
        model.classifier = self.classifier
        
        return model
    
    def forward(self, x):
        eff_features = self.efficientnet(x)
        eff_features = eff_features.unsqueeze(-1).unsqueeze(-1)
        eff_features = self.attention_eff(eff_features)
        eff_features = eff_features.squeeze(-1).squeeze(-1)
        output = self.classifier(eff_features)
        return output
    
    def load_model(self, model_path):
        try:
            logger.info(f"Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            new_state_dict = {}
            
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key.replace('backbone.', 'efficientnet.')
                elif key.startswith('attention.'):
                    new_key = key.replace('attention.', 'attention_eff.')
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            self.model.load_state_dict(new_state_dict, strict=False)
            
            loaded = sum(1 for k in new_state_dict if k in self.model.state_dict())
            total = len(self.model.state_dict())
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"   Loaded {loaded}/{total} layers")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def analyze_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'result': 'ERROR',
                    'confidence': 0,
                    'message': f'Cannot read image: {image_path}',
                    'faces_analyzed': 0
                }
            
            faces = self.face_extractor.extract_faces(image)
            
            if not faces:
                return {
                    'result': 'NO_FACE',
                    'confidence': 0,
                    'message': 'No face detected in image',
                    'faces_analyzed': 0
                }
            
            results = []
            
            for face_data in faces:
                face_tensor = self.face_extractor.preprocess_face(face_data['face'])
                
                with torch.no_grad():
                    outputs = self.forward(face_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][prediction].item()
                
                try:
                    heatmap = self.heatmap_generator.generate_heatmap(
                        self.model, face_tensor, face_data['face']
                    )
                except:
                    heatmap = ""
                
                results.append({
                    'face_id': len(results),
                    'bbox': face_data['bbox'],
                    'prediction': 'FAKE' if prediction == 1 else 'REAL',
                    'confidence': round(confidence * 100, 2),
                    'heatmap': heatmap
                })
            
            fake_confidences = [r['confidence'] for r in results if r['prediction'] == 'FAKE']
            real_confidences = [r['confidence'] for r in results if r['prediction'] == 'REAL']
            
            fake_score = sum(fake_confidences) / len(results) if fake_confidences else 0
            real_score = sum(real_confidences) / len(results) if real_confidences else 0
            
            final_prediction = 'FAKE' if fake_score > real_score else 'REAL'
            final_confidence = max(fake_score, real_score)
            
            return {
                'result': final_prediction,
                'confidence': round(final_confidence, 2),
                'faces_analyzed': len(results),
                'fake_faces': len(fake_confidences),
                'real_faces': len(real_confidences),
                'face_results': results,
                'message': self._generate_message(final_prediction, final_confidence)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                'result': 'ERROR',
                'confidence': 0,
                'message': str(e),
                'faces_analyzed': 0
            }
    
    def _generate_message(self, prediction, confidence):
        if confidence < 60:
            return f"⚠️ Low confidence. Image appears {prediction.lower()}, verify manually."
        elif confidence < 80:
            return f"📊 Medium confidence. Image is likely {prediction.lower()}."
        else:
            return f"✅ High confidence. This image is {prediction.lower()}."
    
    def save_model(self, model_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': 'efficientnet_only'
        }, model_path)
        logger.info(f"✅ Model saved to {model_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing DeepfakeDetector...")
    print("=" * 50)
    
    detector = DeepfakeDetector()
    print(f"✅ Device: {detector.device}")
    print("✅ DeepfakeDetector initialized!")
    print("=" * 50)