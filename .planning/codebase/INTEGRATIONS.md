# INTEGRATIONS.md
Date: 2026-04-19

## External services
- **Hugging Face Hub**: For downloading models (e.g. `tiiuae/Falcon-Perception`, `tiiuae/Falcon-OCR`), datasets (`tiiuae/PBench`).
- **vLLM**: Supports deployment via vLLM Docker Server. 
- **Streamlit**: Used for the demo browser-based application (`demo/streamlit_app.py`).

## External ML Models
- Optionally integrates with PaddlePaddle OCR layout parsers if using `ocr_layout` task.
