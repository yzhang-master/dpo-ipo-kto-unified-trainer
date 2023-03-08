"""Small, explicit preference-learning algorithms implemented in PyTorch."""

from preference_trainer.losses import PAIRWISE_LOSSES, dpo_loss, ipo_loss, kto_loss
from preference_trainer.model import TinyPolicyConfig, TinyPreferencePolicy

__all__ = [
    "PAIRWISE_LOSSES",
    "TinyPolicyConfig",
    "TinyPreferencePolicy",
    "dpo_loss",
    "ipo_loss",
    "kto_loss",
]
