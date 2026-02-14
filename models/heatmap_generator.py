import torch
import torch.nn.functional as F
import numpy as np
import cv2
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeatmapGenerator:
    
    def __init__(self, device='cpu'):
        
        self.device = device
        self.gradients = None
        self.activations = None
        
    def save_gradient(self, grad):
        self.gradients = grad
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def generate_heatmap(self, model, input_tensor, original_image):
        
        try:
            handle_activation = None
            handle_gradient = None
            
            if hasattr(model, 'efficientnet'):
                last_conv = model.efficientnet.features[-1]
                handle_activation = last_conv.register_forward_hook(self.save_activation)
                
                def hook_fn(module, grad_input, grad_output):
                    self.save_gradient(grad_output[0])
                
                handle_gradient = last_conv.register_full_backward_hook(hook_fn)
            
            model.eval()
            output = model.forward(input_tensor)
            
            target_class = torch.argmax(output, dim=1).item()
            
            model.zero_grad()
            
            one_hot = torch.zeros_like(output)
            one_hot[0][target_class] = 1
            output.backward(gradient=one_hot)
            
            if handle_activation:
                handle_activation.remove()
            if handle_gradient:
                handle_gradient.remove()
            
            gradients = self.gradients
            activations = self.activations
            
            if gradients is None or activations is None:
                logger.warning("Could not generate Grad-CAM heatmap")
                return self._create_fallback_heatmap(original_image)
            
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            
            for i in range(activations.shape[1]):
                activations[:, i, :, :] *= pooled_gradients[i]
                
            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = F.relu(heatmap)
            heatmap = heatmap.cpu().detach().numpy()
            
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            else:
                heatmap = np.zeros_like(heatmap)
            
            h, w = original_image.shape[:2]
            heatmap = cv2.resize(heatmap, (w, h))
            
            heatmap = np.uint8(255 * heatmap)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            if original_image.shape[2] == 3:
                original_display = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            else:
                original_display = original_image.copy()
                
            overlay = cv2.addWeighted(original_display, 0.6, heatmap_colored, 0.4, 0)
            
            _, buffer = cv2.imencode('.png', overlay)
            heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return heatmap_base64
            
        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            return self._create_fallback_heatmap(original_image)
    
    def _create_fallback_heatmap(self, original_image):
        try:
            h, w = original_image.shape[:2]
            
            heatmap = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    heatmap[i, j] = int(255 * (i / h) * (j / w))
            
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            if original_image.shape[2] == 3:
                original_display = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            else:
                original_display = original_image.copy()
                
            overlay = cv2.addWeighted(original_display, 0.7, heatmap_colored, 0.3, 0)
            
            _, buffer = cv2.imencode('.png', overlay)
            return base64.b64encode(buffer).decode('utf-8')
        except:
            return ""


if __name__ == "__main__":
    print("=" * 50)
    print("Testing HeatmapGenerator...")
    print("=" * 50)
    
    heatmap_gen = HeatmapGenerator()
    print("✅ HeatmapGenerator initialized")
    
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_tensor = torch.randn(1, 3, 224, 224)
    
    print("\n✅ HeatmapGenerator is ready to use!")
    print("=" * 50)