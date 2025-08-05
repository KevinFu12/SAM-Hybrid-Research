import os
import cv2
import numpy as np
import pandas as pd
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils import compute_dice_iou, compute_hausdorff

# Argument Parser
parser = argparse.ArgumentParser(description="Run SAM Automatic Evaluation")
parser.add_argument("--image_dir", type=str, required=True, help="Path to input images")
parser.add_argument("--mask_dir", type=str, required=True, help="Path to ground truth masks")
parser.add_argument("--output_csv", type=str, default="sam_auto_eval_results.csv", help="Path to save result CSV")
args = parser.parse_args()

# Load SAM
sam_checkpoint = "/content/sam_vit_b.pth"
model_type = "vit_b"
device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    crop_n_layers=0,
    min_mask_region_area=100
)

image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.png'))])
results = []

for img_file in image_files:
    img_path = os.path.join(args.image_dir, img_file)
    mask_path = os.path.join(args.mask_dir, img_file)

    if not os.path.exists(mask_path):
        continue

    # Pre-processing
    # Baca gambar dan normalisasi nilai piksel ke [0, 1]
    image = cv2.imread(img_path).astype(np.float32) / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Baca ground truth mask dan ubah jadi biner (0/1)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_mask > 127).astype(np.uint8)

    # Segmentasi dengan SAM
    masks = mask_generator.generate((image * 255).astype(np.uint8))
    combined_mask = np.zeros_like(gt_mask, dtype=np.uint8)

    for m in masks:
        combined_mask |= m['segmentation'].astype(np.uint8)

    # Post-processing
    # Gabungkan semua mask SAM → lalu binarisasi hasil segmentasi
    pred_mask = (combined_mask >= 1).astype(np.uint8)
    
    # Evaluasi
    dice, iou = compute_dice_iou(pred_mask, gt_mask)
    hausdorff = compute_hausdorff(pred_mask, gt_mask)
    results.append((img_file, dice, iou, hausdorff))

df = pd.DataFrame(results, columns=["Image", "Dice", "IoU", "Hausdorff"])
df.to_csv(args.output_csv, index=False)
