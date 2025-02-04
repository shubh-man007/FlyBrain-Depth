import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

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
    # Convert RGB to grayscale
    guide_gray = np.mean(rgb_img, axis=2)  # Convert to grayscale manually
    return guided_filter(guide_gray, pred_heatmap)

def enhance_sharpness(image, alpha=1.5, beta=-0.5):
    """
    Apply unsharp masking to enhance the sharpness of the image.
    :param image: Input grayscale image.
    :param alpha: Weight for the original image.
    :param beta: Weight for the blurred image.
    :return: Sharpened image.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), 3)  # Gaussian Blur
    sharpened = cv2.addWeighted(image, alpha, blurred, beta, 0)
    return sharpened

def test_and_visualize(model, data_loader, device, save_dir="output"):
    model.eval()
    os.makedirs(f"{save_dir}/rgb_images", exist_ok=True)
    os.makedirs(f"{save_dir}/predicted_depth", exist_ok=True)
    os.makedirs(f"{save_dir}/filtered_depth", exist_ok=True)

    with torch.no_grad():
        for batch_idx, (rgb_tensor, depth_tensor) in enumerate(data_loader):
            rgb_tensor = rgb_tensor.to(device)
            depth_tensor = depth_tensor.to(device).float()
            
            # Run model to get predicted depth
            pred_depth = model(rgb_tensor)  # Shape: [B, 1, H, W]

            # Move to CPU
            pred_depth = pred_depth.cpu().numpy()
            depth_tensor = depth_tensor.cpu().numpy()
            rgb_tensor = rgb_tensor.cpu().numpy()

            # Process first image in batch
            idx = 0
            rgb_img = np.transpose(rgb_tensor[idx], (1, 2, 0))  # H x W x C
            gt_depth = depth_tensor[idx, 0, :, :]
            pred_heatmap = pred_depth[idx, 0, :, :]

            # Normalize for saving
            pred_heatmap_norm = cv2.normalize(pred_heatmap, None, 0, 255, cv2.NORM_MINMAX)
            pred_heatmap_norm = np.uint8(pred_heatmap_norm)

            # Apply Guided Filtering
            refined_depth = apply_guided_filter(rgb_img, pred_heatmap_norm)

            # Enhance Sharpness
            refined_depth_sharp = enhance_sharpness(refined_depth)

            # Save Images
            rgb_save_path = os.path.join(save_dir, "rgb_images", f"rgb_{batch_idx}.png")
            depth_save_path = os.path.join(save_dir, "predicted_depth", f"depth_{batch_idx}.png")
            filtered_save_path = os.path.join(save_dir, "filtered_depth", f"filtered_depth_{batch_idx}.png")

            cv2.imwrite(rgb_save_path, cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(depth_save_path, pred_heatmap_norm)
            cv2.imwrite(filtered_save_path, refined_depth_sharp)

            print(f"Saved: {rgb_save_path}, {depth_save_path}, and {filtered_save_path}")

            # Visualization
            plt.figure(figsize=(24, 6))

            plt.subplot(1, 4, 1)
            plt.imshow(rgb_img)
            plt.title("RGB Image")
            plt.axis("off")

            plt.subplot(1, 4, 2)
            plt.imshow(gt_depth, cmap='viridis')
            plt.title("Ground Truth Depth")
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.imshow(pred_heatmap, cmap='viridis')
            plt.title("Predicted Depth Heatmap")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(refined_depth_sharp, cmap='viridis')
            plt.title("Refined Depth (Guided + Sharpened)")
            plt.axis("off")

            plt.show()

            # Process only one batch
            break


# Example usage:
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_and_visualize(model, test_loader, device)
