# Caspi-1.7B CoreML Conversion

Scripts to convert [OzLabs/Caspi-1.7B](https://huggingface.co/OzLabs/Caspi-1.7B) (Hebrew ASR) from PyTorch to CoreML for on-device inference on Apple Silicon.

Forked from [FluidInference/mobius](https://github.com/FluidInference/mobius/tree/main/models/stt/qwen3-asr-0.6b/coreml) with architecture constants updated for the 1.7B model.

## Related repos

- **CoreML models**: [alandotcom/caspi-1.7b-coreml](https://huggingface.co/alandotcom/caspi-1.7b-coreml) on HuggingFace
- **Hex fork** (macOS dictation app): [alandotcom/Hex](https://github.com/alandotcom/Hex)
- **FluidAudio fork** (inference library): [alandotcom/FluidAudio](https://github.com/alandotcom/FluidAudio)

## Architecture: Caspi-1.7B vs Qwen3-ASR-0.6B

| Component | 0.6B | 1.7B (Caspi) |
|---|---|---|
| Audio encoder layers | 18 | 24 |
| Audio encoder d_model | 896 | 1024 |
| Audio output dim | 1024 | 2048 |
| Decoder hidden size | 1024 | 2048 |
| Decoder intermediate | 2816 | 6144 |
| Decoder layers | 28 | 28 |
| Attention heads | 16 | 16 |
| KV heads | 8 | 8 |
| Head dim | 128 | 128 |
| Vocab size | 151,936 | 151,936 |

## Prerequisites

- macOS with Apple Silicon
- Python 3.10+ with [uv](https://docs.astral.sh/uv/)
- ~10 GB disk space for weights + converted models

## Setup

```bash
git clone https://github.com/alandotcom/caspi-hebrew-asr.git
cd caspi-hebrew-asr

# Clone dependencies (needed by conversion scripts)
git clone https://github.com/FluidInference/mobius.git
git clone https://github.com/QwenLM/Qwen3-ASR.git qwen3-asr

# Install Python deps
cd conversion
uv sync
```

## Conversion

### Full pipeline (produces 5-model .mlpackage set)

```bash
cd conversion
uv run python convert-qwen3-asr.py
```

### Fused stateful decoder (what FluidAudio uses)

```bash
uv run python convert_decoder_fused.py --model-id OzLabs/Caspi-1.7B --output-dir build/caspi-1.7b
```

### Extract embeddings + vocab

```bash
uv run python extract_embeddings.py --model-id OzLabs/Caspi-1.7B --output-dir build/caspi-1.7b
```

### Quantize to int8

```bash
uv run python quantize_model.py build/caspi-1.7b/qwen3_asr_decoder_stateful.mlpackage build/caspi-1.7b-int8/qwen3_asr_decoder_stateful.mlpackage --dtype int8
```

### Final model directory (for FluidAudio)

```
caspi-1.7b-coreml/
  qwen3_asr_audio_encoder_v2.mlmodelc/
  qwen3_asr_decoder_stateful.mlmodelc/
  qwen3_asr_embeddings.bin
  vocab.json
```

Place in `~/Library/Application Support/FluidAudio/Models/caspi-1.7b-coreml/` or let Hex auto-download from HuggingFace.

## Performance (M5 Pro, 48GB)

| Variant | Decoder size | Peak memory | RTFx | Hebrew WER |
|---|---|---|---|---|
| f32 | 3.2 GB | ~11 GB | 2.2x | ~5% |
| int8 | 1.6 GB | ~6 GB | 2.1x | ~5% |

## License

Conversion scripts: Apache-2.0 (from [FluidInference/mobius](https://github.com/FluidInference/mobius))

Caspi model weights: CC-BY-NC-4.0 (from [OzLabs/Caspi-1.7B](https://huggingface.co/OzLabs/Caspi-1.7B))
