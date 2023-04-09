import argparse
import copy
import json
import platform
import time
from pathlib import Path

import torch

from preference_trainer.cache import (
    ReferenceLogprobCache,
    build_reference_cache,
    reference_token_count,
)
from preference_trainer.data import preference_datasets
from preference_trainer.model import (
    CharTokenizer,
    TinyPolicyConfig,
    TinyPreferencePolicy,
    cached_forward_flops,
    matrix_forward_flops_per_token,
)
from preference_trainer.trainer import evaluate_preference_pairs, train_preferences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["dpo", "ipo", "kto"], default="dpo")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke", action="store_true", help="Run two epochs on four pairs")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    tokenizer = CharTokenizer()
    model = TinyPreferencePolicy(TinyPolicyConfig(tokenizer.vocab_size))
    reference = copy.deepcopy(model).eval()
    train_pairs, eval_pairs = preference_datasets()
    if args.smoke:
        train_pairs = train_pairs[:4]
        args.epochs = 2

    output = args.output or Path(f"results/{args.method}_training.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = output.with_name(f"{args.method}_reference_cache.pt")
    cache_start = time.perf_counter()
    cache = build_reference_cache(reference, tokenizer, train_pairs)
    cache_seconds = time.perf_counter() - cache_start
    cache.save(cache_path)
    cache = ReferenceLogprobCache.load(cache_path)
    baseline = evaluate_preference_pairs(model, tokenizer, eval_pairs)
    result = train_preferences(
        model,
        tokenizer,
        train_pairs,
        cache,
        method=args.method,
        beta=args.beta,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
    final = evaluate_preference_pairs(model, tokenizer, eval_pairs)
    train_evaluation = evaluate_preference_pairs(model, tokenizer, train_pairs)
    tokens = reference_token_count(tokenizer, train_pairs)
    per_token = matrix_forward_flops_per_token(model.config)
    torch.save(model.state_dict(), output.with_name(f"{args.method}_policy.pt"))

    report = {
        "scope": "tiny randomly initialized character GRU; not an LLM quality benchmark",
        "method": args.method,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "baseline_eval": baseline,
        "final_eval": final,
        "final_train": train_evaluation,
        "initial_loss": result.history[0]["loss"],
        "final_loss": result.final_loss,
        "loss_history": result.history,
        "training_seconds": result.training_seconds,
        "reference_cache": {
            "path": str(cache_path),
            "unique_responses": len(cache.entries),
            "build_seconds": cache_seconds,
            "full_forward_tokens": tokens,
            "matrix_flops_per_token": per_token,
            "analytical_flops_saved_per_epoch": cached_forward_flops(model.config, tokens),
            "analytical_flops_saved_total": cached_forward_flops(
                model.config, tokens, args.epochs
            ),
            "note": (
                "Analytical matrix multiply-add FLOPs only; excludes elementwise ops "
                "and memory traffic."
            ),
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {key: value for key, value in report.items() if key != "loss_history"}
    print(json.dumps(summary, indent=2))
    print(f"Saved full report to {output}")


if __name__ == "__main__":
    main()
