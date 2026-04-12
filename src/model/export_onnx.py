"""Export script to convert Keras (.h5 / .keras) models to ONNX format."""

import argparse
import logging
from pathlib import Path

import tensorflow as tf
import tf2onnx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CloudShadow-UNet to ONNX")
    parser.add_argument("--model", type=Path, required=True, help="Path to the input Keras model (.h5 or .keras)")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output ONNX model (.onnx)")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version (default: 13)")
    return parser.parse_args()

def export_to_onnx(model_path: Path, output_path: Path, opset: int = 13) -> None:
    """Loads a Keras model and exports it via tf2onnx."""
    
    if not model_path.exists():
        raise FileNotFoundError(f"Source model not found: {model_path}")
    
    logger.info(f"Loading Keras model from {model_path}...")
    
    # We load without compiling since inference doesn't need the optimizer or custom losses
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Assuming input is [Batch, H, W, 4] where H,W can be None for dynamic sizing
    input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input_tensor")]
    
    logger.info(f"Converting model to ONNX (opset={opset})...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    onnx_model, _ = tf2onnx.convert.from_keras(
        model, 
        input_signature=input_signature, 
        opset=opset, 
        output_path=str(output_path)
    )
    
    logger.info(f"ONNX model successfully saved to: {output_path}")

if __name__ == "__main__":
    args = parse_args()
    export_to_onnx(args.model, args.output, args.opset)
