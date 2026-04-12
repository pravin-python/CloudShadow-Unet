import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Sen2Cor Wrapper for Level-1C to Level-2A")
    parser.add_argument("--l1c_dir", type=Path, required=True, help="Directory containing .SAFE L1C product folders")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory for L2A products")
    parser.add_argument("--resolution", type=int, choices=[10, 20, 60], default=10, help="Target resolution in metres")
    return parser.parse_args()

def run_sen2cor(l1c_safe: Path, out_dir: Path, resolution: int = 10) -> bool:
    """Invokes Sen2Cor subprocess on a single .SAFE folder."""
    logger.info(f"Invoking L2A_Process for: {l1c_safe.name}")
    
    cmd = [
        "L2A_Process", 
        str(l1c_safe), 
        "--resolution", str(resolution)
    ]
    
    if out_dir is not None:
        cmd.extend(["--output_dir", str(out_dir)])
        
    try:
        # Run subprocess and stream output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            # We log at debug to avoid flooding, unless there's an error
            logger.debug(line.strip())
            
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"Successfully processed: {l1c_safe.name}")
            return True
        else:
            logger.error(f"Sen2Cor failed for {l1c_safe.name} with return code {process.returncode}")
            return False
            
    except FileNotFoundError:
        logger.error("Sen2Cor executable 'L2A_Process' not found in PATH. Please install Sen2Cor.")
        return False

def process_directory(l1c_dir: Path, out_dir: Path, resolution: int):
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        
    safe_dirs = list(l1c_dir.glob("*.SAFE"))
    if not safe_dirs:
        logger.warning(f"No .SAFE product folders found in {l1c_dir}")
        return
        
    success_count = 0
    for safe in safe_dirs:
        if run_sen2cor(safe, out_dir, resolution):
            success_count += 1
            
    logger.info(f"Done. Processed {success_count}/{len(safe_dirs)} products.")

if __name__ == "__main__":
    args = parse_args()
    process_directory(args.l1c_dir, args.out_dir, args.resolution)
