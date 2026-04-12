"""
Active Learning Loop:
Runs inference on a pool of unannotated images, scores patches by prediction uncertainty (entropy),
and clips out the top N most uncertain patches into a separate directory for manual human review/annotation.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import tensorflow as tf

from src.preprocessing.preprocess import read_multiband_geotiff

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Active Learning Miner")
    parser.add_argument("--unlabelled_dir", type=Path, required=True)
    parser.add_argument("--review_dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--top_n", type=int, default=50, help="Number of uncertain patches to save per scene")
    return parser.parse_args()

def calculate_entropy(softmax_probs: np.ndarray) -> np.ndarray:
    """Calculate Shannon Entropy along the channel axis (H, W, C)."""
    # Clip to avoid log(0)
    p = np.clip(softmax_probs, 1e-7, 1.0)
    entropy = -np.sum(p * np.log(p), axis=-1)
    return entropy

def mine_uncertainty(model, unlabelled_dir: Path, review_dir: Path, patch_size: int, top_n: int):
    review_dir.mkdir(parents=True, exist_ok=True)
    
    tif_files = list(unlabelled_dir.glob("*.tif*"))
    for t_path in tif_files:
        logger.info(f"Scanning {t_path.name}...")
        image, profile = read_multiband_geotiff(t_path)
        h, w = image.shape[:2]
        
        # Grid based patching for simplicity in active learning
        patch_info = []
        
        for r in range(0, h - patch_size + 1, patch_size):
            for c in range(0, w - patch_size + 1, patch_size):
                patch = image[r:r+patch_size, c:c+patch_size, :]
                patch_info.append({
                    'r': r, 'c': c, 'patch': patch
                })
        
        if not patch_info:
            continue
            
        patches = np.stack([p['patch'] for p in patch_info], axis=0)
        
        # Run inference in one batch or loop if memory constrained
        # For simplicity, doing it in loop to save memory:
        batch_size = 16
        uncertainties = []
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            probs = model.predict(batch, verbose=0)
            entropy = calculate_entropy(probs)
            # Average entropy per patch
            mean_entropy = np.mean(entropy, axis=(1, 2))
            uncertainties.extend(mean_entropy)
            
        # Get top N uncertain indices
        top_indices = np.argsort(uncertainties)[-top_n:][::-1]
        
        # Save patches for review
        with rasterio.open(t_path) as src:
            for idx in top_indices:
                info = patch_info[idx]
                r, c = info['r'], info['c']
                unc = uncertainties[idx]
                
                # We need to write the un-normalized raw patch (or a visualizer rgb) for review
                # We will write out a cropped tiff window from the original source.
                window = Window(c, r, patch_size, patch_size)
                transform = src.window_transform(window)
                raw_patch = src.read(window=window)
                
                out_profile = src.profile.copy()
                out_profile.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": transform
                })
                
                out_path = review_dir / f"{t_path.stem}_r{r}_c{c}_unc{unc:.3f}.tif"
                with rasterio.open(out_path, "w", **out_profile) as dest:
                    dest.write(raw_patch)
                    
        logger.info(f"Saved {top_n} patches to {review_dir}")

def main():
    args = parse_args()
    
    # Custom objects handling
    from src.model.losses import CombinedLoss, DiceCoefficient, MeanIoU
    logger.info("Loading model...")
    model = tf.keras.models.load_model(
        str(args.model),
        custom_objects={
            "CombinedLoss": CombinedLoss,
            "DiceCoefficient": DiceCoefficient,
            "MeanIoU": MeanIoU,
        },
    )
    
    mine_uncertainty(model, args.unlabelled_dir, args.review_dir, args.patch_size, args.top_n)

if __name__ == "__main__":
    main()
