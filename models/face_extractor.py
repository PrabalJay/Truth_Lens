import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceExtractor:
    
    def __init__(self, device=None):
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"Initializing FaceExtractor on {self.device}")
        
        self.face_detector = MTCNN(
            keep_all=True,
            device=self.device,
            select_largest=True,
            post_process=False,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7]
        )
    
    def extract_faces(self, image, confidence_threshold=0.95):
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image[0,0,0] > image[0,0,2]:  
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image.copy()
        else:
            image_rgb = image
        
        boxes, probabilities = self.face_detector.detect(image_rgb)
        
        faces = []
        
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probabilities)):
                if prob >= confidence_threshold:
                    x1, y1, x2, y2 = box.astype(int)
                    
                    margin_h = int((x2 - x1) * 0.2)
                    margin_v = int((y2 - y1) * 0.2)
                    
                    x1 = max(0, x1 - margin_h)
                    y1 = max(0, y1 - margin_v)
                    x2 = min(image_rgb.shape[1], x2 + margin_h)
                    y2 = min(image_rgb.shape[0], y2 + margin_v)
                    
                    face = image_rgb[y1:y2, x1:x2]
                    
                    try:
                        face_resized = cv2.resize(face, (224, 224))
                    except:
                        continue
                    
                    faces.append({
                        'face': face_resized,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': prob,
                        'original_image': image_rgb
                    })
            
            logger.info(f"Detected {len(faces)} faces")
        else:
            logger.info("No faces detected")
            
        return faces
    
    def extract_face_single(self, image):
        
        faces = self.extract_faces(image)
        
        if faces:
            largest = max(faces, key=lambda x: 
                        (x['bbox'][2] - x['bbox'][0]) * 
                        (x['bbox'][3] - x['bbox'][1]))
            return largest
        
        return None
    
    def preprocess_face(self, face_image):
        
        face_image = face_image.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        face_image = (face_image - mean) / std
        
        face_tensor = torch.from_numpy(face_image).permute(2, 0, 1).float()
        
        face_tensor = face_tensor.unsqueeze(0)
        
        return face_tensor.to(self.device)


if __name__ == "__main__":
    print("=" * 50)
    print("Testing FaceExtractor...")
    print("=" * 50)
    
    extractor = FaceExtractor()
    print(f"✅ Device: {extractor.device}")
    
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    faces = extractor.extract_faces(dummy_image)
    print(f"✅ Face extraction test complete")
    
    print("\n✅ FaceExtractor is ready to use!")
    print("=" * 50)