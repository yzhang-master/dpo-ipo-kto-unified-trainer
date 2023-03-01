import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_beta(beta: float) -> None:
    if beta <= 0:
        raise ValueError("beta must be positive")


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor,
    reference_rejected_logps: Tensor,
    beta: float = 0.1,
) -> Tensor:
    """Direct Preference Optimization loss for a batch of preference pairs."""
    _validate_beta(beta)
    logits = beta * (
        (policy_chosen_logps - policy_rejected_logps)
        - (reference_chosen_logps - reference_rejected_logps)
    )
    return -F.logsigmoid(logits).mean()


def pairwise_accuracy(policy_chosen_logps: Tensor, policy_rejected_logps: Tensor) -> Tensor:
    """Return the fraction of pairs whose chosen completion has higher policy logprob."""
    return (policy_chosen_logps > policy_rejected_logps).to(torch.float32).mean()
