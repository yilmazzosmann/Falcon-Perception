# STRUCTURE.md
Date: 2026-04-19

## Directory Layout
- `falcon_perception/`: Core package folder.
  - `__init__.py`, `__main__.py`
  - `model.py`, `attention.py`, `kv_cache.py`, `paged_attention.py`, `rope.py`: Core Transformer implementations.
  - `anyup.py`, `aux_output.py`: Vision/Upsampling heads.
  - `paged_inference.py`, `paged_ocr_inference.py`, `batch_inference.py`: Runner/Generation Engines.
  - `data.py`: Preprocessing, tokenization.
  - `mlx/`: Apple Silicon implementations.
  - `server/`: FastAPI server implementations.
- `demo/`: Scripts to test model, notebooks (`perception.ipynb`, `ocr.ipynb`), benchmark runners.
- `eval/`: Scripts to evaluate metric benchmarks (e.g. `pbench.py`, `metrics.py`).
- `pyproject.toml`: Dependency tracking and project entry point.
- `README.md`: Central documentation.
