# dpo-ipo-kto-unified-trainer

> **Era:** 2023–2024 · Senior AI Engineer focus area (preference optimization). Techniques from 2023, reimplemented in 2026 with current PyTorch.

A unified, CPU-testable preference trainer with pluggable DPO, IPO, and binary-outcome KTO losses, offline reference-logprob caching, a genuinely trainable character GRU, deterministic preference data, and pairwise win-rate evaluation. Everything runs locally without `transformers`, `datasets`, hosted models, or API keys.

## Problem

Preference optimization repeatedly needs both the trainable policy and a fixed reference policy. Recomputing the unchanged reference on every epoch wastes work, while combining objectives often obscures their different supervision and formulas. This repository makes the three losses explicit, tests them against independent calculations, and measures both preference learning and analytical reference-compute savings on a tiny real model.

## Method

For completion logprobs `π(y | x)`, reference logprobs `πref(y | x)`, chosen/rejected responses `yw, yl`, and inverse temperature `β`, the pairwise objectives are:

```text
Δ = (log π(yw | x) - log π(yl | x))
    - (log πref(yw | x) - log πref(yl | x))

DPO = mean[-log sigmoid(β Δ)]
IPO = mean[(Δ - 1 / (2β))²]
```

KTO instead accepts independent desirable/undesirable labels. With response log-ratio `r = log π(y | x) - log πref(y | x)` and detached batch anchor `KL̂ = stopgrad(mean(r))`:

```text
Ldesirable   = 1 - sigmoid(β (r - KL̂))
Lundesirable = 1 - sigmoid(β (KL̂ - r))
KTO = λd mean(Ldesirable) + λu mean(Lundesirable)
```

Group means are computed only for labels present in the batch. The KTO anchor is detached so it centers the utility functions without adding a separate direct KL-minimization gradient.

The policy is a 7,042-parameter character-level GRU (16-dimensional embeddings, 24 hidden units, 97-token fixed ASCII vocabulary). Teacher-forced logprobs sum only completion tokens, not prompt or padding tokens. The fixed dataset has 24 training pairs and eight disjoint evaluation pairs containing simple factual preferences.

Before training, `scripts/train.py` deep-copies the initial policy, computes every unique training response's reference logprob, and stores values under SHA-256 keys of `[prompt, completion]`. It saves and reloads that cache before the first optimizer step. Cache misses fail explicitly. Training reads these tensors and performs no reference forward pass. Saved caches include a fingerprint of the reference weights and round-trip through `torch.load(..., weights_only=True)`.

Analytical savings count two FLOPs per multiply-add in the GRU input/recurrent matrices and output projection:

```text
FLOPs/token = 6 H (E + H) + 2 H V = 10,464
saved = FLOPs/token × unique full-forward tokens × epochs
```

This is an operation count, **not** measured latency or energy. It excludes nonlinearities, embeddings, memory traffic, and cache serialization.

## Results

Measured locally with Python 3.12.3, Torch 2.15.0+cpu, seed 7, one Torch thread, 60 full-batch epochs, learning rate 0.01, and `β=0.1`. The same eight evaluation pairs are used for every method. Win rate compares summed chosen and rejected completion logprobs; ties count as half a win. These are single deterministic runs, not statistical confidence intervals.

| Objective | Initial loss | Final loss | Train win rate | Eval win rate, before → after | Final eval margin | Training time |
|---|---:|---:|---:|---:|---:|---:|
| DPO | 0.693147 | 0.186673 | 87.50% | 37.50% → 37.50% | -19.428791 | 2.182 s |
| IPO | 25.000000 | 0.182182 | 75.00% | 37.50% → 37.50% | -2.774976 | 2.391 s |
| KTO | 1.000000 | 0.407844 | 83.33% | 37.50% → 50.00% | -18.144058 | 1.559 s |

The negative evaluation margins and lack of DPO/IPO held-out improvement are retained rather than presented as successful generalization.

The final benchmark driver took 25.960 seconds wall time, including three fresh Python/Torch process starts. Each run saved and reloaded 48 responses covering 1,570 forward tokens. At 60 epochs, each objective avoided an analytical **985,708,800 FLOPs (0.986 GFLOP)** of reference matrix work. Cache-build times were 0.017 s (DPO), 0.016 s (IPO), and 0.016 s (KTO).

These numbers describe a randomly initialized toy GRU on eight tiny held-out pairs. They are **not LLM win rates, human preferences, reward-model results, or evidence that one objective is generally better**.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check .
pytest
python scripts/train.py --smoke --output results/smoke_training.json
python scripts/benchmark.py
```

Without installing the package, prefix Python commands with `PYTHONPATH=src`. Individual objectives can be run with:

```bash
python scripts/train.py --method dpo --epochs 60 --output results/dpo_training.json
python scripts/train.py --method ipo --epochs 60 --output results/ipo_training.json
python scripts/train.py --method kto --epochs 60 --output results/kto_training.json
```

The JSON reports include every epoch's loss, exact win/tie counts, train/eval margins, timings, cache metadata, and analytical savings. Tests cover independent loss formulas, backpropagation, KTO anchor detachment, teacher forcing, cache misses/round-trips, fixed data, FLOP arithmetic, real parameter updates, and deterministic training.

## Limitations

- The model is trained from random initialization on synthetic facts; it is not a pretrained LM and does not support open-ended generation or human evaluation.
- Eight evaluation pairs are far too few for reliable comparisons. DPO and KTO substantially overfit by epoch 60, as their negative held-out margins show.
- Full-batch training is deliberate for reproducibility, not a scalable data pipeline. There is no packing, distributed training, mixed precision, or checkpoint-resume logic.
- The KTO implementation uses a detached mean response log-ratio as its batch KL anchor. This is the usual finite-batch surrogate, not an exact sum over the full next-token distribution.
- Analytical FLOPs do not establish a wall-clock speedup. A large-model cached/uncached latency comparison remains future work.
- No GPU, external corpus, hosted LLM, or API evaluation was used or claimed.

## References

- R. Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023. arXiv:2305.18290
- M. G. Azar et al. *A General Theoretical Paradigm to Understand Learning from Human Preferences.* AISTATS 2024. arXiv:2310.12036 (IPO / ΨPO)
- K. Ethayarajh et al. *KTO: Model Alignment as Prospect-Theoretic Optimization.* ICML 2024. arXiv:2402.01306

## License

MIT
