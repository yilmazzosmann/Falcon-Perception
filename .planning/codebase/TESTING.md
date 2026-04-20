# TESTING.md
Date: 2026-04-19

## Framework & Tools
- No prominent unit-test frameworks like `pytest` found at the root or `tests/`.
- Testing is effectively performed through the `eval/` structure (e.g., `pbench.py`, `metrics.py`) which acts as integration testing logic against core benchmarks.
- Verification is also handled through interactive `demo/` notebooks and single run scripts.

## Coverage
- No explicit coverage reports found. It focuses primarily on benchmark evaluation (mAP, IoU logic).
