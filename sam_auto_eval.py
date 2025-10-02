import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import SAM_CHECKPOINT, SAM_MODEL_TYPE, device, DICE_SUCCESS_THRESHOLD
from sam_segmenter import SAMSegmenter
from evaluator import Evaluator
from utils import load_image_mask

def main():
    parser = argparse.ArgumentParser(description="Run SAM Automatic Evaluation")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to input images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to ground truth masks")
    parser.add_argument("--output_dir", type=str, default="./results_sam_auto", help="Path to save results")
    args = parser.parse_args()

    # Initialize components
    sam_segmenter = SAMSegmenter()
    evaluator = Evaluator()

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.png'))])

    if len(image_files) == 0:
        print(f"No images found in {args.image_dir}")
        return

    print(f"Processing {len(image_files)} images...")

    for img_file in tqdm(image_files, desc="SAM Auto Segmentation"):
        try:
            # Load image and mask
            image_path = os.path.join(args.image_dir, img_file)
            mask_path = os.path.join(args.mask_dir, img_file)
            
            if not os.path.exists(mask_path):
                print(f"Warning: Ground truth mask not found for {img_file}, skipping...")
                continue
            
            image_np, gt_mask = load_image_mask(image_path, mask_path)
            
            # SAM auto-masking
            sam_segmenter.set_image((image_np * 255).astype(np.uint8))
            pred_mask = sam_segmenter.auto_mask()
            
            # Evaluation
            dice, iou, hausdorff, status = evaluator.evaluate_segmentation(pred_mask, gt_mask)
            
            # Save predicted mask
            output_path = os.path.join(args.output_dir, img_file.replace(".jpg", "_mask.png").replace(".png", "_mask.png"))
            cv2.imwrite(output_path, pred_mask * 255)

            results.append({
                "Image": img_file,
                "Dice": dice,
                "IoU": iou,
                "Hausdorff": hausdorff,
                "Status": status
            })

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            results.append({
                "Image": img_file,
                "Dice": 0.0,
                "IoU": 0.0,
                "Hausdorff": np.inf,
                "Status": "ERROR"
            })
            continue

    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "sam_auto_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"SAM Auto Evaluation Complete")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total images processed: {len(df)}")
    print(f"Average Dice: {df['Dice'].mean():.3f}")
    print(f"Average IoU: {df['IoU'].mean():.3f}")
    print(f"Average Hausdorff: {df['Hausdorff'].replace([np.inf], np.nan).mean():.2f}")
    print(f"Successful segmentations (Dice >= {DICE_SUCCESS_THRESHOLD}): {len(df[df['Status'] == 'OK'])}/{len(df)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()