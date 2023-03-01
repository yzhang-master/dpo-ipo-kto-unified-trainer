import torch

from preference_trainer.losses import dpo_loss, pairwise_accuracy


def test_dpo_matches_manual_logsigmoid() -> None:
    policy_chosen = torch.tensor([-1.2, -3.0])
    policy_rejected = torch.tensor([-2.0, -1.5])
    reference_chosen = torch.tensor([-1.0, -2.5])
    reference_rejected = torch.tensor([-1.8, -2.0])
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
