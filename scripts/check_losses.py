import torch

from preference_trainer.losses import dpo_loss


def main() -> None:
    chosen = torch.tensor([-0.8], requires_grad=True)
    rejected = torch.tensor([-1.4], requires_grad=True)
    reference = torch.tensor([-1.0])
    loss = dpo_loss(chosen, rejected, reference, reference)
    loss.backward()
    print(f"DPO smoke loss: {loss.item():.6f}")
    print(f"chosen gradient: {chosen.grad.item():.6f}")


if __name__ == "__main__":
    main()
