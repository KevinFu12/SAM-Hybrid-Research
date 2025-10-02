import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from config import SAM_CHECKPOINT, SAM_MODEL_TYPE, device

class SAMSegmenter:
    """Wrapper for SAM segmentation with box prompts and auto-masking"""
    
    def __init__(self):
        """Initialize SAM model"""
        self.sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device)
        self.predictor = SamPredictor(self.sam)
        
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=100
        )
        
        self.current_image = None
    
    def set_image(self, image):
        """Set image for SAM predictor
        
        Args:
            image: numpy array of shape (H, W, 3) with values in [0, 255]
        """
        self.current_image = image
        self.predictor.set_image(image)
    
    def predict_from_boxes(self, boxes, scores=None, plural_flag=False):
        """Generate segmentation mask from detected boxes
        
        Args:
            boxes: tensor or numpy array of shape (N, 4) in format [x1, y1, x2, y2]
            scores: optional confidence scores for each box
            plural_flag: if True, combine all boxes; if False, use highest confidence box
            
        Returns:
            Binary mask (H, W) as numpy array
        """
        if self.current_image is None:
            raise ValueError("Must call set_image() before predict_from_boxes()")
        
        # Convert boxes to numpy if needed
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        
        if len(boxes) == 0:
            return self.auto_mask()
        
        # Handle single vs multiple wounds
        if plural_flag or len(boxes) > 1:
            # Combine all detected boxes
            combined_mask = np.zeros(
                (self.current_image.shape[0], self.current_image.shape[1]), 
                dtype=np.uint8
            )
            
            for box in boxes:
                masks, _, _ = self.predictor.predict(
                    box=box,
                    multimask_output=False
                )
                combined_mask = np.logical_or(combined_mask, masks[0]).astype(np.uint8)
            
            return combined_mask
        else:
            # Use single highest confidence box
            box_to_use = boxes[0] if scores is None else boxes[np.argmax(scores)]
            masks, _, _ = self.predictor.predict(
                box=box_to_use,
                multimask_output=False
            )
            return masks[0].astype(np.uint8)
    
    def auto_mask(self):
        """Generate mask using SAM's automatic mask generation
        
        Returns:
            Binary mask (H, W) as numpy array
        """
        if self.current_image is None:
            raise ValueError("Must call set_image() before auto_mask()")
        
        masks = self.mask_generator.generate(self.current_image)
        
        if len(masks) == 0:
            return np.zeros(
                (self.current_image.shape[0], self.current_image.shape[1]), 
                dtype=np.uint8
            )
        
        # Combine all auto-generated masks
        combined_mask = np.zeros(
            (self.current_image.shape[0], self.current_image.shape[1]), 
            dtype=np.uint8
        )
        
        for mask_data in masks:
            combined_mask = np.logical_or(
                combined_mask, 
                mask_data['segmentation']
            ).astype(np.uint8)
        
        return combined_mask