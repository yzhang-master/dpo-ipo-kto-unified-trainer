import pytest
import torch

from preference_trainer.losses import (
    PAIRWISE_LOSSES,
    dpo_loss,
    ipo_loss,
    kto_loss,
    pairwise_accuracy,
)


def pair_inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([-1.2, -3.0]),
        torch.tensor([-2.0, -1.5]),
        torch.tensor([-1.0, -2.5]),
        torch.tensor([-1.8, -2.0]),
    )


def test_dpo_matches_manual_logsigmoid() -> None:
    policy_chosen, policy_rejected, reference_chosen, reference_rejected = pair_inputs()
    beta = 0.2

    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    manual_logits = beta * (policy_margin - reference_margin)
    expected = -torch.log(torch.sigmoid(manual_logits)).mean()

    assert torch.allclose(
        dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta),
        expected,
    )


def test_dpo_is_log_two_at_reference_and_backpropagates() -> None:
    reference_chosen = torch.tensor([-1.0, -2.0])
    reference_rejected = torch.tensor([-1.5, -1.0])
    chosen = reference_chosen.clone().requires_grad_(True)
    rejected = reference_rejected.clone().requires_grad_(True)

    loss = dpo_loss(chosen, rejected, reference_chosen, reference_rejected)
    loss.backward()

    assert torch.allclose(loss, torch.log(torch.tensor(2.0)))
    assert chosen.grad is not None
    assert rejected.grad is not None
    assert torch.allclose(chosen.grad, -rejected.grad)
    assert pairwise_accuracy(chosen.detach(), rejected.detach()).item() == 0.5


def test_ipo_matches_manual_squared_margin() -> None:
    policy_chosen, policy_rejected, reference_chosen, reference_rejected = pair_inputs()
    beta = 0.25
    margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    expected = ((margin - 2.0) ** 2).mean()

    assert torch.allclose(
        ipo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta),
        expected,
    )


def test_ipo_is_zero_at_target_margin() -> None:
    reference = torch.zeros(2)
    chosen = torch.tensor([2.0, 2.0])
    rejected = torch.zeros(2)
    loss = ipo_loss(chosen, rejected, reference, reference, beta=0.25)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_kto_matches_independent_group_formula() -> None:
    policy = torch.tensor([-0.5, -1.5, -3.0])
    reference = torch.tensor([-1.0, -1.0, -2.0])
    desirable = torch.tensor([True, True, False])
    beta = 0.2
    ratio = policy - reference
    kl = ratio.mean().detach()
    expected_desirable = (1 - torch.sigmoid(beta * (ratio[:2] - kl))).mean()
    expected_undesirable = 1 - torch.sigmoid(beta * (kl - ratio[2]))
    expected = 2.0 * expected_desirable + 0.5 * expected_undesirable

    actual = kto_loss(
        policy,
        reference,
        desirable,
        beta=beta,
        desirable_weight=2.0,
        undesirable_weight=0.5,
    )
    assert torch.allclose(actual, expected)


def test_kto_kl_anchor_is_detached_and_gradients_reach_policy() -> None:
    reference = torch.tensor([-1.0, -1.0, -2.0])
    labels = torch.tensor([True, True, False])
    policy = torch.tensor([-0.5, -2.0, -3.0], requires_grad=True)
    detached_loss = kto_loss(policy, reference, labels)
    detached_loss.backward()

    attached_policy = torch.tensor([-0.5, -2.0, -3.0], requires_grad=True)
    ratio = attached_policy - reference
    kl = ratio.mean()
    desirable_loss = (1 - torch.sigmoid(0.1 * (ratio[:2] - kl))).mean()
    undesirable_loss = 1 - torch.sigmoid(0.1 * (kl - ratio[2]))
    (desirable_loss + undesirable_loss).backward()

    assert policy.grad is not None
    assert not torch.allclose(policy.grad, torch.zeros_like(policy))
    assert not torch.allclose(policy.grad, attached_policy.grad)


@pytest.mark.parametrize("method", sorted(PAIRWISE_LOSSES))
def test_pairwise_losses_reject_nonpositive_beta(method: str) -> None:
    tensors = pair_inputs()
    with pytest.raises(ValueError, match="beta"):
        PAIRWISE_LOSSES[method](*tensors, beta=0.0)
