# Caspi-1.7B CoreML Conversion

Converts OzLabs/Caspi-1.7B (Hebrew fine-tune of Qwen3-ASR-1.7B) to CoreML for on-device Apple Silicon inference.

Forked from [FluidInference/mobius](https://github.com/FluidInference/mobius/tree/main/models/stt/qwen3-asr-0.6b/coreml) with dimension constants updated for the 1.7B architecture.

## Usage

```bash
uv sync
uv run python convert-qwen3-asr.py
uv run python convert-qwen3-asr.py --components audio_encoder  # single component
```

## Source models

- Base: [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (Apache-2.0)
- Fine-tune: [OzLabs/Caspi-1.7B](https://huggingface.co/OzLabs/Caspi-1.7B) (CC-BY-NC-4.0)
