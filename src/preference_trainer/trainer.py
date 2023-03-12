import time
from dataclasses import dataclass

import torch
from torch import Tensor

from preference_trainer.cache import ReferenceLogprobCache
from preference_trainer.data import PreferencePair
from preference_trainer.losses import PAIRWISE_LOSSES, kto_loss
from preference_trainer.model import CharTokenizer, TinyPreferencePolicy, sequence_logps


@dataclass
class TrainingResult:
    method: str
    epochs: int
    final_loss: float
    history: list[dict[str, float]]
    training_seconds: float


def _cached_pair_logps(
    cache: ReferenceLogprobCache, pairs: list[PreferencePair]
) -> tuple[Tensor, Tensor]:
    chosen = torch.stack([cache.get(pair.prompt, pair.chosen) for pair in pairs])
    rejected = torch.stack([cache.get(pair.prompt, pair.rejected) for pair in pairs])
    return chosen, rejected


def train_preferences(
    model: TinyPreferencePolicy,
    tokenizer: CharTokenizer,
    pairs: list[PreferencePair],
    cache: ReferenceLogprobCache,
    method: str = "dpo",
    beta: float = 0.1,
    learning_rate: float = 0.01,
    epochs: int = 40,
) -> TrainingResult:
    """Full-batch preference training with no reference forward in the loop."""
    if method not in {"dpo", "ipo", "kto"}:
        raise ValueError(f"unknown preference objective: {method}")
    if not pairs or epochs <= 0 or learning_rate <= 0:
        raise ValueError("pairs and epochs must be non-empty/positive and learning_rate positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if method == "kto":
            prompts = [pair.prompt for pair in pairs for _ in range(2)]
            completions = [response for pair in pairs for response in (pair.chosen, pair.rejected)]
            labels = torch.tensor([label for pair in pairs for label in (True, False)])
            reference = torch.stack(
                [
                    cache.get(prompt, completion)
                    for prompt, completion in zip(prompts, completions, strict=True)
                ]
            )
            policy = sequence_logps(model, tokenizer, prompts, completions)
            loss = kto_loss(policy, reference, labels, beta=beta)
        else:
            chosen = sequence_logps(
                model, tokenizer, [pair.prompt for pair in pairs], [pair.chosen for pair in pairs]
            )
            rejected = sequence_logps(
                model, tokenizer, [pair.prompt for pair in pairs], [pair.rejected for pair in pairs]
            )
            reference_chosen, reference_rejected = _cached_pair_logps(cache, pairs)
            loss = PAIRWISE_LOSSES[method](
                chosen, rejected, reference_chosen, reference_rejected, beta=beta
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        history.append({"epoch": float(epoch + 1), "loss": float(loss.detach())})

    return TrainingResult(
        method=method,
        epochs=epochs,
        final_loss=history[-1]["loss"],
        history=history,
        training_seconds=time.perf_counter() - start,
    )


@torch.no_grad()
def evaluate_preference_pairs(
    model: TinyPreferencePolicy,
    tokenizer: CharTokenizer,
    pairs: list[PreferencePair],
) -> dict[str, float | int]:
    """Score pairwise win rate; an exact logprob tie counts as half a win."""
    model.eval()
    chosen = sequence_logps(
        model, tokenizer, [pair.prompt for pair in pairs], [pair.chosen for pair in pairs]
    )
    rejected = sequence_logps(
        model, tokenizer, [pair.prompt for pair in pairs], [pair.rejected for pair in pairs]
    )
    wins = int((chosen > rejected).sum())
    ties = int((chosen == rejected).sum())
    return {
        "pairs": len(pairs),
        "wins": wins,
        "ties": ties,
        "win_rate": (wins + 0.5 * ties) / len(pairs),
        "mean_chosen_logp": float(chosen.mean()),
        "mean_rejected_logp": float(rejected.mean()),
        "mean_margin": float((chosen - rejected).mean()),
    }
