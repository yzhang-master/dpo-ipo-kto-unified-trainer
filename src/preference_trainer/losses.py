from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_beta(beta: float) -> None:
    if beta <= 0:
        raise ValueError("beta must be positive")


def _validate_pair_shapes(*tensors: Tensor) -> None:
    if not tensors or any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("logprobs must be one-dimensional batches")
    if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
        raise ValueError("all logprob batches must have the same shape")


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor,
    reference_rejected_logps: Tensor,
    beta: float = 0.1,
) -> Tensor:
    """Direct Preference Optimization loss for a batch of preference pairs."""
    _validate_beta(beta)
    _validate_pair_shapes(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    logits = beta * (
        (policy_chosen_logps - policy_rejected_logps)
        - (reference_chosen_logps - reference_rejected_logps)
    )
    return -F.logsigmoid(logits).mean()


def ipo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor,
    reference_rejected_logps: Tensor,
    beta: float = 0.1,
) -> Tensor:
    """Identity Preference Optimization loss with target margin 1 / (2 beta)."""
    _validate_beta(beta)
    _validate_pair_shapes(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    margin = (policy_chosen_logps - policy_rejected_logps) - (
        reference_chosen_logps - reference_rejected_logps
    )
    return ((margin - 1.0 / (2.0 * beta)) ** 2).mean()


def kto_loss(
    policy_logps: Tensor,
    reference_logps: Tensor,
    desirable: Tensor,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> Tensor:
    """Kahneman-Tversky Optimization loss for binary outcome labels.

    The KL anchor is estimated from the current batch's response log-ratio and
    detached, as in the reference KTO implementation. Desirable and undesirable
    examples contribute their own weighted group means.
    """
    _validate_beta(beta)
    _validate_pair_shapes(policy_logps, reference_logps)
    if desirable.shape != policy_logps.shape or desirable.dtype != torch.bool:
        raise ValueError("desirable must be a bool tensor matching the logprob batch")
    if desirable_weight <= 0 or undesirable_weight <= 0:
        raise ValueError("KTO group weights must be positive")

    log_ratio = policy_logps - reference_logps
    kl_anchor = log_ratio.mean().detach()
    desirable_losses = 1.0 - torch.sigmoid(beta * (log_ratio - kl_anchor))
    undesirable_losses = 1.0 - torch.sigmoid(beta * (kl_anchor - log_ratio))

    group_losses = []
    if desirable.any():
        group_losses.append(desirable_weight * desirable_losses[desirable].mean())
    if (~desirable).any():
        group_losses.append(undesirable_weight * undesirable_losses[~desirable].mean())
    if not group_losses:
        raise ValueError("KTO requires a non-empty batch")
    return torch.stack(group_losses).sum()


def pairwise_accuracy(policy_chosen_logps: Tensor, policy_rejected_logps: Tensor) -> Tensor:
    """Return the fraction of pairs whose chosen completion has higher policy logprob."""
    _validate_pair_shapes(policy_chosen_logps, policy_rejected_logps)
    return (policy_chosen_logps > policy_rejected_logps).to(torch.float32).mean()


PAIRWISE_LOSSES: dict[str, Callable[..., Tensor]] = {"dpo": dpo_loss, "ipo": ipo_loss}
