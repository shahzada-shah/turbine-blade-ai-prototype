from PIL import Image, ImageDraw
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2

def create_dummy_heatmap(pil_image, output_path):
    """Fallback dummy heatmap when model not available"""
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    radius = min(w, h) // 4
    bbox = (w//2 - radius, h//2 - radius, w//2 + radius, h//2 + radius)
    draw.ellipse(bbox, outline="red", width=10)
    os.makedirs("results", exist_ok=True)
    img.save(output_path)
    return output_path


def create_gradcam_heatmap(model, tensor, pil_image, output_path, target_class_idx=None):
    """
    Create Grad-CAM heatmap visualization
    
    Args:
        model: PyTorch model (must be in eval mode)
        tensor: Preprocessed input tensor (batch_size=1)
        pil_image: Original PIL image for overlay
        output_path: Path to save heatmap
        target_class_idx: Class index to visualize (None = use predicted class)
    """
    if model is None:
        return create_dummy_heatmap(pil_image, output_path)
    
    try:
        device = next(model.parameters()).device
        tensor = tensor.to(device)
        
        # Register hooks for activations and gradients
        activations = {}
        gradients = {}
        
        def forward_hook(module, input, output):
            activations['layer4'] = output
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients['layer4'] = grad_output[0]
        
        # Use layer4 (last conv block) for ResNet50
        target_layer = model.layer4
        handle_forward = target_layer.register_forward_hook(forward_hook)
        try:
            handle_backward = target_layer.register_full_backward_hook(backward_hook)
        except:
            # Fallback for older PyTorch versions
            handle_backward = target_layer.register_backward_hook(backward_hook)
        
        # Forward pass (need gradients enabled)
        model.eval()
        tensor.requires_grad_()
        output = model(tensor)
        
        # Get target class (predicted if not specified)
        if target_class_idx is None:
            target_class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        model.zero_grad()
        target = output[0, target_class_idx]
        target.backward()
        
        # Get gradients and activations
        if 'layer4' not in gradients or 'layer4' not in activations:
            # Fallback if hooks didn't capture properly
            return create_dummy_heatmap(pil_image, output_path)
        
        grads = gradients['layer4']
        acts = activations['layer4']
        
        if grads is None or acts is None:
            return create_dummy_heatmap(pil_image, output_path)
        
        # Global average pooling of gradients
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU
        cam = F.interpolate(cam, size=(pil_image.size[1], pil_image.size[0]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to 0-1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Convert to heatmap colormap
        cam_uint8 = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert PIL to numpy for blending
        img_np = np.array(pil_image)
        if img_np.shape[2] == 4:  # RGBA
            img_np = img_np[:, :, :3]  # Remove alpha
        
        # Resize heatmap to match image
        if heatmap.shape[:2] != img_np.shape[:2]:
            heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        
        # Blend heatmap with original image
        alpha = 0.4  # Transparency factor
        blended = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
        
        # Remove hooks
        try:
            handle_forward.remove()
            handle_backward.remove()
        except:
            pass
        
        # Save result
        os.makedirs("results", exist_ok=True)
        result_img = Image.fromarray(blended)
        result_img.save(output_path)
        
        return output_path
    except Exception as e:
        # Fallback to dummy heatmap on any error
        print(f"Grad-CAM failed (falling back to dummy): {e}")
        return create_dummy_heatmap(pil_image, output_path)
