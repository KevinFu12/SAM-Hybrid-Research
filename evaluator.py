import numpy as np
from utils import compute_dice_iou, compute_hausdorff
from config import DICE_SUCCESS_THRESHOLD

class Evaluator:
    @staticmethod
    def compute_dice(pred_mask, gt_mask):
        dice, _ = compute_dice_iou(pred_mask, gt_mask)
        return dice
    
    @staticmethod
    def compute_iou(pred_mask, gt_mask):
        _, iou = compute_dice_iou(pred_mask, gt_mask)
        return iou
    
    @staticmethod
    def compute_hausdorff(pred_mask, gt_mask):
        return compute_hausdorff(pred_mask, gt_mask)
    
    @staticmethod
    def evaluate_segmentation(pred_mask, gt_mask):
        """Comprehensive evaluation of segmentation results"""
        dice, iou = compute_dice_iou(pred_mask, gt_mask)
        hausdorff = compute_hausdorff(pred_mask, gt_mask)
        status = "OK" if dice >= DICE_SUCCESS_THRESHOLD else "FN"
        
        return dice, iou, hausdorff, status