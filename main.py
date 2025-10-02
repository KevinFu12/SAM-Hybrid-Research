import argparse
import subprocess
import sys
from config import print_config

def run_sam_auto(image_dir, mask_dir, output_dir):
    """Run SAM automatic segmentation evaluation"""
    print("\n" + "="*60)
    print("Running SAM Auto Evaluation")
    print("="*60 + "\n")
    
    cmd = [
        "python3", "sam_auto_eval.py",
        "--image_dir", image_dir,
        "--mask_dir", mask_dir,
        "--output_dir", output_dir
    ]
    
    result = subprocess.run(cmd)
    return result.returncode

def run_hybrid(image_dir, mask_dir, output_dir, max_attempts=3):
    """Run Hybrid GPT-4o + OWL-ViT + SAM Evaluation"""
    print("\n" + "="*60)
    print("Running Hybrid GPT-4o + OWL-ViT + SAM Evaluation")
    print("="*60 + "\n")
    
    cmd = [
        "python3", "sam_hybrid_eval.py",
        "--image_dir", image_dir,
        "--gt_mask_dir", mask_dir,
        "--output_dir", output_dir,
        "--max_attempts", str(max_attempts)
    ]
    
    result = subprocess.run(cmd)
    return result.returncode

def run_comparison(sam_auto_results, hybrid_results):
    """Compare results from both methods"""
    print("\n" + "="*60)
    print("Comparing SAM Auto vs Hybrid Results")
    print("="*60 + "\n")
    
    try:
        import pandas as pd
        
        df_sam = pd.read_csv(sam_auto_results)
        df_hybrid = pd.read_csv(hybrid_results)
        
        print(f"{'Metric':<20} {'SAM Auto':>15} {'Hybrid':>15} {'Difference':>15}")
        print("-"*65)
        
        avg_dice_sam = df_sam['Dice'].mean()
        avg_dice_hybrid = df_hybrid['Dice'].mean()
        print(f"{'Average Dice':<20} {avg_dice_sam:>15.3f} {avg_dice_hybrid:>15.3f} {avg_dice_hybrid-avg_dice_sam:>+15.3f}")
        
        avg_iou_sam = df_sam['IoU'].mean()
        avg_iou_hybrid = df_hybrid['IoU'].mean()
        print(f"{'Average IoU':<20} {avg_iou_sam:>15.3f} {avg_iou_hybrid:>15.3f} {avg_iou_hybrid-avg_iou_sam:>+15.3f}")
        
        success_sam = len(df_sam[df_sam['Status'] == 'OK'])
        success_hybrid = len(df_hybrid[df_hybrid['Status'] == 'OK'])
        print(f"{'Successful (OK)':<20} {success_sam:>15} {success_hybrid:>15} {success_hybrid-success_sam:>+15}")
        
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        print(f"Error comparing results: {e}")
        return 1
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Run segmentation evaluation pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run SAM auto-masking only
  python main.py sam_auto --image_dir ./images --mask_dir ./masks
  
  # Run hybrid pipeline with custom output
  python main.py hybrid --image_dir ./images --mask_dir ./masks --output_dir ./results_hybrid
  
  # Compare results from both methods
  python main.py compare --sam_results ./results_sam/sam_auto_results.csv --hybrid_results ./results_hybrid/hybrid_results.csv
  
  # Show configuration
  python main.py config
        """
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Choose which pipeline to run")
    
    # SAM Auto mode
    parser_sam = subparsers.add_parser("sam_auto", help="Run SAM automatic segmentation")
    parser_sam.add_argument("--image_dir", type=str, required=True, help="Path to input images")
    parser_sam.add_argument("--mask_dir", type=str, required=True, help="Path to ground truth masks")
    parser_sam.add_argument("--output_dir", type=str, default="./results_sam_auto", help="Output directory")
    
    # Hybrid mode
    parser_hybrid = subparsers.add_parser("hybrid", help="Run hybrid GPT-4o + OWL-ViT + SAM")
    parser_hybrid.add_argument("--image_dir", type=str, required=True, help="Path to input images")
    parser_hybrid.add_argument("--mask_dir", type=str, required=True, help="Path to ground truth masks")
    parser_hybrid.add_argument("--output_dir", type=str, default="./results_hybrid", help="Output directory")
    parser_hybrid.add_argument("--max_attempts", type=int, default=3, help="Max GPT-4o attempts per image")
    
    # Compare mode
    parser_compare = subparsers.add_parser("compare", help="Compare SAM auto vs hybrid results")
    parser_compare.add_argument("--sam_results", type=str, required=True, help="Path to SAM auto results CSV")
    parser_compare.add_argument("--hybrid_results", type=str, required=True, help="Path to hybrid results CSV")
    
    # Config mode
    parser_config = subparsers.add_parser("config", help="Show current configuration")
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        sys.exit(1)
    
    if args.mode == "sam_auto":
        return run_sam_auto(args.image_dir, args.mask_dir, args.output_dir)
    
    elif args.mode == "hybrid":
        return run_hybrid(args.image_dir, args.mask_dir, args.output_dir, args.max_attempts)
    
    elif args.mode == "compare":
        return run_comparison(args.sam_results, args.hybrid_results)
    
    elif args.mode == "config":
        print_config()
        return 0

if __name__ == "__main__":
    sys.exit(main())