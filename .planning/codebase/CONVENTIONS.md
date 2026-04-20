# CONVENTIONS.md
Date: 2026-04-19

## Code Style
- Follows modern Python type hinting practices (PEP 484).
- Uses `uv` as the default package manager and dependency indexer.
- Dependency segregation via optional packages `[torch]`, `[mlx]`, `[dev]`, `[server]`, and `[ocr]`.

## Patterns
- Supports automatic backend fallbacks (`torch` vs `mlx`) based on platform (`sys_platform` == `darwin` and `platform_machine` == `arm64`).
- Highly modularized attention modules with `flex_attention` to utilize fused Triton kernels.
