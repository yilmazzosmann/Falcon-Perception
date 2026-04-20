# STACK.md
Date: 2026-04-19

## Stack
- **Languages**: Python (>= 3.10)
- **Frameworks/Runtime**: PyTorch (>= 2.11) natively with CUDA 12.8 / 13.0, MLX (>= 0.30.0) for Apple Silicon
- **Web/API**: FastAPI, Uvicorn, Streamlit
- **Build System**: setuptools, uv

## Modules/Dependencies
- **Core ML libraries**: einops, numpy, pillow, tokenizers, opencv-python, scipy, safetensors, transformers
- **HuggingFace integrations**: datasets, hf-transfer, hf-xet
- **CLI/Config**: tyro

## Architecture / Hardware Target
- NVIDIA GPUs with CUDA
- Apple M-series with MLX
