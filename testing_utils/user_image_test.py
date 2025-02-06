import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.ndimage
from torchvision import transforms

def guided_filter(I, p, radius=64, eps=1e-3):
    """
    Perform guided filtering to refine the depth map.
    :param I: Guidance image (grayscale)
    :param p: Input image to be filtered (depth map)
    :param radius: Window radius
    :param eps: Regularization term
    :return: Filtered image
    """
    mean_I = scipy.ndimage.uniform_filter(I, radius)
    mean_p = scipy.ndimage.uniform_filter(p, radius)
    corr_I = scipy.ndimage.uniform_filter(I * I, radius)
    corr_Ip = scipy.ndimage.uniform_filter(I * p, radius)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = scipy.ndimage.uniform_filter(a, radius)
    mean_b = scipy.ndimage.uniform_filter(b, radius)

    q = mean_a * I + mean_b
    return q

def apply_guided_filter(rgb_img, pred_heatmap):
    """
    Use the RGB image (converted to grayscale) as guidance to refine the depth map.
    """
    guide_gray = np.mean(rgb_img, axis=2)  
    return guided_filter(guide_gray, pred_heatmap)

def enhance_sharpness(image, alpha=1.5, beta=-0.5):
    """
    Enhance image sharpness using unsharp masking.
    :param image: Input image (grayscale)
    :param alpha: Weight for the original image.
    :param beta: Weight for the blurred image.
    :return: Sharpened image.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, alpha, blurred, beta, 0)
    return sharpened

def load_and_preprocess_image(image_path, img_height=480, img_width=640):
    """
    Load an image from disk, resize, normalize, and convert to a torch tensor.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))  
    img_tensor = img_tensor.unsqueeze(0)  
    
    return img_tensor, img  


def test_image_path(model, device, image_path, save_dir="output", img_height=480, img_width=640):
    """
    Load an image from a given path, run the model to predict depth, and apply guided filtering
    and sharpness enhancement to refine the depth map.
    Saves and visualizes the results.
    """
    os.makedirs(os.path.join(save_dir, "rgb_images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "predicted_depth"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "filtered_depth"), exist_ok=True)
    
    img_tensor, rgb_img = load_and_preprocess_image(image_path, img_height, img_width)
    img_tensor = img_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        pred_depth = model(img_tensor)
    
    pred_depth = torch.nn.functional.interpolate(pred_depth, size=(img_height, img_width), mode='bilinear', align_corners=True)
    
    pred_depth_np = pred_depth.squeeze().cpu().numpy()  
    
    pred_depth_norm = cv2.normalize(pred_depth_np, None, 0, 255, cv2.NORM_MINMAX)
    pred_depth_norm = np.uint8(pred_depth_norm)
    
    refined_depth = apply_guided_filter(rgb_img, pred_depth_norm)
    
    refined_depth_sharp = enhance_sharpness(refined_depth)
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    rgb_save_path = os.path.join(save_dir, "rgb_images", f"{base_filename}_rgb.png")
    depth_save_path = os.path.join(save_dir, "predicted_depth", f"{base_filename}_depth.png")
    filtered_save_path = os.path.join(save_dir, "filtered_depth", f"{base_filename}_filtered.png")
    
    cv2.imwrite(rgb_save_path, cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(depth_save_path, pred_depth_norm)
    cv2.imwrite(filtered_save_path, refined_depth_sharp)
    
    print(f"Saved: {rgb_save_path}, {depth_save_path}, and {filtered_save_path}")
    
    plt.figure(figsize=(24, 6))
    
    plt.subplot(1, 4, 1)
    plt.imshow(rgb_img)
    plt.title("RGB Image")
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.imshow(pred_depth_np, cmap='viridis')
    plt.title("Predicted Depth Heatmap")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(refined_depth, cmap='viridis')
    plt.title("Refined Depth (Guided Filter)")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.imshow(refined_depth_sharp, cmap='viridis')
    plt.title("Refined Depth (Sharpness Enhanced)")
    plt.axis("off")
    
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_image_file = "IMAGE_PATH" 
test_image_path(model, device, test_image_file, save_dir="output", img_height=480, img_width=640)
