"""
Quantitative Evaluation Script for Cloud/Shadow Segmentation.
Compares U-Net predictions and other algorithms (FMask, Sen2Cor, etc.) 
against ground truth masks and outputs tabular metrics (IoU, Dice).
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Benchmark Cloud/Shadow Models")
    parser.add_argument("--truth_dir", type=Path, required=True, help="Directory with ground truth masks")
    parser.add_argument("--pred_dirs", type=Path, nargs="+", required=True, help="Directories with model predictions")
    parser.add_argument("--labels", type=str, nargs="+", required=True, help="Names of the models/algorithms")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark_results.csv"))
    return parser.parse_args()

def safe_divide(a, b):
    return a / b if b != 0 else 0.0

def compute_metrics(truth_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int = 3):
    """Computes IoU and Dice Coefficient per class."""
    metrics = {}
    for c in range(num_classes):
        t = (truth_mask == c)
        p = (pred_mask == c)
        intersection = np.logical_and(t, p).sum()
        union = np.logical_or(t, p).sum()
        
        iou = safe_divide(intersection, union)
        dice = safe_divide(2 * intersection, t.sum() + p.sum())
        
        c_name = {0: "Bg", 1: "Cloud", 2: "Shadow"}[c]
        metrics[f"{c_name}_IoU"] = iou
        metrics[f"{c_name}_Dice"] = dice
        
    metrics["Mean_IoU"] = np.mean([metrics[f"{x}_IoU"] for x in ["Bg", "Cloud", "Shadow"]])
    return metrics

def run_benchmark(truth_dir: Path, pred_dirs: list[Path], labels: list[str], output_csv: Path):
    if len(pred_dirs) != len(labels):
        raise ValueError("Number of pred_dirs must match number of labels.")
        
    truth_files = sorted(truth_dir.glob("*.tif*"))
    if not truth_files:
        logger.warning(f"No truth mask files found in {truth_dir}")
        return
        
    results = []
    
    for t_idx, t_path in enumerate(truth_files):
        with rasterio.open(t_path) as src:
            truth = src.read(1)
            
        scene_name = t_path.stem
        
        for p_dir, label in zip(pred_dirs, labels):
            p_path = p_dir / t_path.name
            if not p_path.exists():
                logger.warning(f"Missing prediction for {scene_name} in {p_dir}")
                continue
                
            with rasterio.open(p_path) as src:
                pred = src.read(1)
                
            # Compute metrics
            m = compute_metrics(truth, pred)
            m["Algorithm"] = label
            m["Scene"] = scene_name
            results.append(m)
            
    if not results:
        logger.error("No results computed.")
        return
        
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ["Algorithm", "Scene", "Bg_IoU", "Bg_Dice", "Cloud_IoU", "Cloud_Dice", "Shadow_IoU", "Shadow_Dice", "Mean_IoU"]
    df = df[cols]
    
    # Save raw
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    # Print summary
    summary = df.groupby("Algorithm").mean(numeric_only=True).round(4)
    logger.info("\n=== Benchmark Summary ===")
    logger.info(summary.to_string())
    logger.info(f"\nResults saved to {output_csv}")

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args.truth_dir, args.pred_dirs, args.labels, args.output)
