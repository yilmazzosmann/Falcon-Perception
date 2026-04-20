# ARCHITECTURE.md
Date: 2026-04-19

## Pattern
- **Transformers/Vision-Language Models**: Uses a unified Transformer architecture for vision-language tasks (segmentation, detection, OCR).
- **Inference Engines**: 
  - **Paged Inference Engine** (`paged_inference.py`, `paged_ocr_inference.py`): Continuous batching via paged KV cache, preemption, high-resolution image cache, CUDA graphs support (FlexAttention).
  - **Batch Inference Engine** (`batch_inference.py`): Dense KV cache, left-padded batch inference.
  - **MLX Engine**: Native implementation in `falcon_perception/mlx/` folder for Apple Silicon without PyTorch dependency.
  
## Data Flow
- Standard LLM autoregressive token decoding but supports multimodel inputs. Image tokens and text tokens are processed bidirectionally/causally respectively.

## Layers
- `falcon_perception/server/` for FastAPI/HTTP REST API layers.
- Core logic (`model.py`, `paged_attention.py`, `anyup.py`) manages PyTorch network definition.
- `data.py` data processing loops and tokenizer prep.
