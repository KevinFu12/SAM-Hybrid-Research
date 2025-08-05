import argparse
import subprocess

def run_sam_auto():
    print("Running SAM Auto Evaluation...")
    subprocess.run(["python3", "sam_auto_eval.py"])

def run_hybrid():
    print("Running Hybrid GPT-4o + OWL-ViT + SAM Evaluation...")
    subprocess.run(["python3", "sam_hybrid_eval.py"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run segmentation evaluation modules.")
    parser.add_argument("mode", choices=["sam_auto", "hybrid", "manual"], help="Choose which pipeline to run.")
    args = parser.parse_args()

    if args.mode == "sam_auto":
        run_sam_auto()
    elif args.mode == "hybrid":
        run_hybrid()
