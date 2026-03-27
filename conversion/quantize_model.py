#!/usr/bin/env python3
"""Apply int8 quantization to a CoreML .mlpackage."""
import argparse
from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .mlpackage path")
    parser.add_argument("output", help="Output .mlpackage path")
    parser.add_argument("--dtype", default="int8", choices=["int8", "int4"])
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    model = ct.models.MLModel(args.input)

    print(f"Quantizing to {args.dtype}...")
    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(dtype=args.dtype)
    )
    quantized = linear_quantize_weights(model, config=config)

    print(f"Saving to {args.output}...")
    quantized.save(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
