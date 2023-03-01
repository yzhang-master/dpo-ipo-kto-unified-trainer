# dpo-ipo-kto-unified-trainer

> **Era:** 2023–2024 · Senior AI Engineer focus area (preference optimization). Techniques from 2023, reimplemented in 2026 with current PyTorch.

A local, CPU-testable implementation of Direct Preference Optimization. IPO, KTO, a tiny trainable policy, offline reference caching, and measured training results are added in subsequent implementation stages.

## Reproduce

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

No hosted model, external dataset, API key, or GPU is required.

## References

- R. Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023. arXiv:2305.18290

## License

MIT
