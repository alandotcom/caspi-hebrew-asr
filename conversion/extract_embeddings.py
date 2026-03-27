#!/usr/bin/env python3
"""Extract embedding weights and vocab for FluidAudio's 2-model pipeline.

Produces:
  - qwen3_asr_embeddings.bin: [uint32 vocab_size, uint32 hidden_size, float16 data...]
  - vocab.json: {token_id: token_string, ...} (int->string mapping)

Usage:
  uv run python extract_embeddings.py --model-id OzLabs/Caspi-1.7B --output-dir build/caspi-1.7b
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings and vocab for FluidAudio")
    parser.add_argument("--model-id", default="OzLabs/Caspi-1.7B", help="HuggingFace model ID")
    parser.add_argument("--output-dir", default="build/caspi-1.7b", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # Load embedding weights from safetensors
    print(f"Loading model weights from {args.model_id}...")
    st_path = hf_hub_download(args.model_id, "model.safetensors")
    weights = load_file(st_path)

    # Find embedding weights (thinker.model.embed_tokens.weight)
    embed_key = "thinker.model.embed_tokens.weight"
    if embed_key not in weights:
        print(f"Key '{embed_key}' not found. Available keys with 'embed':")
        for k in weights:
            if "embed" in k.lower():
                print(f"  {k}: {weights[k].shape}")
        return

    embed_weight = weights[embed_key].float().numpy()
    vocab_size, hidden_size = embed_weight.shape
    print(f"Embedding shape: {embed_weight.shape} (vocab_size={vocab_size}, hidden_size={hidden_size})")

    # Write binary: [uint32 vocab_size, uint32 hidden_size, float16 data...]
    embed_fp16 = embed_weight.astype(np.float16)
    bin_path = output_dir / "qwen3_asr_embeddings.bin"
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<II", vocab_size, hidden_size))
        f.write(embed_fp16.tobytes())

    expected_size = 8 + vocab_size * hidden_size * 2
    actual_size = bin_path.stat().st_size
    print(f"Saved embeddings: {bin_path} ({actual_size:,} bytes, expected {expected_size:,})")
    assert actual_size == expected_size, f"Size mismatch!"

    # Download and convert vocab.json
    # FluidAudio expects {token_id_int: token_string} mapping
    print(f"\nDownloading vocab.json...")
    vocab_path = hf_hub_download(args.model_id, "vocab.json")
    with open(vocab_path) as f:
        vocab_data = json.load(f)

    # The HF vocab.json is {token_string: token_id} — FluidAudio needs {token_id: token_string}
    if isinstance(next(iter(vocab_data.values())), int):
        # It's string->int, invert it
        inverted = {v: k for k, v in vocab_data.items()}
        print(f"Inverted vocab: {len(inverted)} entries (string->int to int->string)")
        vocab_out = inverted
    else:
        vocab_out = vocab_data

    out_vocab_path = output_dir / "vocab.json"
    with open(out_vocab_path, "w") as f:
        json.dump(vocab_out, f, ensure_ascii=False)
    print(f"Saved vocab: {out_vocab_path} ({len(vocab_out)} entries)")

    del weights  # Free memory
    print("\nDone!")


if __name__ == "__main__":
    main()
