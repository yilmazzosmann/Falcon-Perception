# CONCERNS.md
Date: 2026-04-19

## Tech Debt / Fragile Areas
- **Sub-dependencies & Environments**: Hard dependency on very specific, newer CUDA versions (12.8 / 13.0) and latest PyTorch versions (2.11+) due to FlexAttention optimizations. Needs tightly controlled installation environments.
- No automated unit test suite.
- Relies heavily on hardware-specific kernels (Triton / MLX).

## Missing Functionality
- Potential missing coverage around edge-cases in continuous batching/paged KV cache due to lack of comprehensive unit testing.
