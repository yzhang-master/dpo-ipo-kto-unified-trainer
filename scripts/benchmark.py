import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for method in ("dpo", "ipo", "kto"):
        command = [
            sys.executable,
            "scripts/train.py",
            "--method",
            method,
            "--epochs",
            str(args.epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--beta",
            str(args.beta),
            "--seed",
            str(args.seed),
            "--output",
            str(args.output.with_name(f"{method}_training.json")),
        ]
        if args.smoke:
            command.append("--smoke")
        subprocess.run(command, check=True)

    reports = [
        json.loads(args.output.with_name(f"{method}_training.json").read_text())
        for method in ("dpo", "ipo", "kto")
    ]
    summary = {
        "scope": reports[0]["scope"],
        "wall_seconds": time.perf_counter() - started,
        "results": [
            {
                "method": report["method"],
                "epochs": report["epochs"],
                "initial_loss": report["initial_loss"],
                "final_loss": report["final_loss"],
                "baseline_win_rate": report["baseline_eval"]["win_rate"],
                "final_eval_win_rate": report["final_eval"]["win_rate"],
                "final_eval_margin": report["final_eval"]["mean_margin"],
                "final_train_win_rate": report["final_train"]["win_rate"],
                "training_seconds": report["training_seconds"],
                "cache_build_seconds": report["reference_cache"]["build_seconds"],
                "analytical_flops_saved_total": report["reference_cache"][
                    "analytical_flops_saved_total"
                ],
            }
            for report in reports
        ],
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
