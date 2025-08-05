import numpy as np
from scipy.spatial.distance import directed_hausdorff
import cv2

def compute_dice_iou(pred, true):
    pred, true = pred.astype(bool), true.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    dice = 2 * intersection / (pred.sum() + true.sum() + 1e-6)
    iou = intersection / (union + 1e-6)
    return dice, iou

def compute_hausdorff(mask1, mask2):
    points1 = np.argwhere(mask1)
    points2 = np.argwhere(mask2)
    if points1.size == 0 or points2.size == 0:
        return np.inf
    hd1 = directed_hausdorff(points1, points2)[0]
    hd2 = directed_hausdorff(points2, points1)[0]
    return max(hd1, hd2)
