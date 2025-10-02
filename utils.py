import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff

def compute_dice_iou(pred, gt):
    """Compute Dice coefficient and IoU
    
    Args:
        pred: Predicted binary mask (H, W)
        gt: Ground truth binary mask (H, W)
        
    Returns:
        tuple: (dice, iou) scores
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    dice = (2. * intersection) / (pred.sum() + gt.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    
    return dice, iou

def compute_hausdorff(pred, gt):
    """Compute Hausdorff distance between two binary masks
    
    Args:
        pred: Predicted binary mask (H, W)
        gt: Ground truth binary mask (H, W)
        
    Returns:
        float: Hausdorff distance (infinity if either mask is empty)
    """
    if pred.sum() == 0 or gt.sum() == 0:
        return np.inf
    
    try:
        pred_coords = np.column_stack(np.where(pred > 0))
        gt_coords = np.column_stack(np.where(gt > 0))
        
        hausdorff_dist = max(
            directed_hausdorff(pred_coords, gt_coords)[0],
            directed_hausdorff(gt_coords, pred_coords)[0]
        )
        
        return hausdorff_dist
    except Exception as e:
        print(f"Warning: Hausdorff computation failed: {e}")
        return np.inf

def load_image_mask(image_path, mask_path):
    """Load and preprocess image and mask
    
    Args:
        image_path: Path to input image
        mask_path: Path to ground truth mask
        
    Returns:
        tuple: (image_np, gt_mask) where image is normalized to [0,1] and mask is binary
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")
    
    gt_mask = (mask > 127).astype(np.uint8)
    
    return image, gt_mask

def visualize_segmentation(image, pred_mask, gt_mask=None, alpha=0.5):
    """Visualize segmentation result with overlay
    
    Args:
        image: Input image (H, W, 3) normalized to [0, 1]
        pred_mask: Predicted mask (H, W)
        gt_mask: Optional ground truth mask (H, W)
        alpha: Transparency for overlay
        
    Returns:
        numpy array: Visualization image
    """
    # Convert image to uint8
    vis_image = (image * 255).astype(np.uint8).copy()
    
    # Create colored overlay for prediction (green)
    pred_overlay = np.zeros_like(vis_image)
    pred_overlay[pred_mask > 0] = [0, 255, 0]
    
    # Blend prediction
    vis_image = cv2.addWeighted(vis_image, 1 - alpha, pred_overlay, alpha, 0)
    
    # Add ground truth overlay (red) if provided
    if gt_mask is not None:
        gt_overlay = np.zeros_like(vis_image)
        gt_overlay[gt_mask > 0] = [255, 0, 0]
        vis_image = cv2.addWeighted(vis_image, 1 - alpha, gt_overlay, alpha, 0)
        
        # Highlight overlap (yellow)
        overlap = np.logical_and(pred_mask > 0, gt_mask > 0)
        vis_image[overlap] = [255, 255, 0]
    
    return vis_image

def save_comparison_image(image_path, pred_mask, gt_mask, output_path):
    """Save side-by-side comparison of prediction and ground truth
    
    Args:
        image_path: Path to original image
        pred_mask: Predicted mask
        gt_mask: Ground truth mask
        output_path: Path to save comparison image
    """
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0
    
    # Create visualizations
    pred_vis = visualize_segmentation(image_normalized, pred_mask, alpha=0.4)
    gt_vis = visualize_segmentation(image_normalized, gt_mask, alpha=0.4)
    overlay_vis = visualize_segmentation(image_normalized, pred_mask, gt_mask, alpha=0.4)
    
    # Concatenate horizontally
    comparison = np.hstack([image_rgb, pred_vis, gt_vis, overlay_vis])
    
    # Add labels
    h, w = comparison.shape[:2]
    label_height = 30
    labeled = np.ones((h + label_height, w, 3), dtype=np.uint8) * 255
    labeled[label_height:, :, :] = comparison
    
    # Add text labels
    labels = ["Original", "Prediction", "Ground Truth", "Overlay"]
    section_width = w // 4
    for i, label in enumerate(labels):
        cv2.putText(labeled, label, (i * section_width + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Save
    cv2.imwrite(output_path, cv2.cvtColor(labeled, cv2.COLOR_RGB2BGR))

def compute_all_metrics(pred_mask, gt_mask):
    """Compute all evaluation metrics
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        dict: Dictionary containing all metrics
    """
    dice, iou = compute_dice_iou(pred_mask, gt_mask)
    hausdorff = compute_hausdorff(pred_mask, gt_mask)
    
    # Additional metrics
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
    tn = np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'dice': dice,
        'iou': iou,
        'hausdorff': hausdorff,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }