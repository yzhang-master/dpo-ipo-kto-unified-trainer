import copy

import pytest
import torch

from preference_trainer.cache import build_reference_cache
from preference_trainer.data import preference_datasets
from preference_trainer.model import CharTokenizer, TinyPolicyConfig, TinyPreferencePolicy
from preference_trainer.trainer import evaluate_preference_pairs, train_preferences


def setup_training():
    torch.manual_seed(11)
    tokenizer = CharTokenizer()
    model = TinyPreferencePolicy(TinyPolicyConfig(tokenizer.vocab_size, 8, 12))
    pairs = preference_datasets()[0][:4]
    cache = build_reference_cache(model, tokenizer, pairs)
    return tokenizer, model, pairs, cache


@pytest.mark.parametrize("method", ["dpo", "ipo", "kto"])
def test_real_training_updates_policy_and_preserves_cache(method: str) -> None:
    tokenizer, model, pairs, cache = setup_training()
    original_entries = dict(cache.entries)
    before = copy.deepcopy(model.state_dict())

    result = train_preferences(
        model, tokenizer, pairs, cache, method=method, epochs=2, learning_rate=0.01
    )

    assert result.method == method
    assert len(result.history) == 2
    assert result.final_loss >= 0
    assert result.training_seconds > 0
    assert any(not torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())
    assert cache.entries == original_entries


def test_training_is_deterministic_with_fixed_seed() -> None:
    tokenizer, model, pairs, cache = setup_training()
    second = copy.deepcopy(model)
    first_result = train_preferences(model, tokenizer, pairs, cache, epochs=3)
    second_result = train_preferences(second, tokenizer, pairs, cache, epochs=3)
    assert first_result.history == second_result.history
    assert first_result.final_loss == second_result.final_loss
    assert evaluate_preference_pairs(model, tokenizer, pairs)["pairs"] == 4


def test_unknown_objective_is_rejected() -> None:
    tokenizer, model, pairs, cache = setup_training()
    with pytest.raises(ValueError, match="unknown"):
        train_preferences(model, tokenizer, pairs, cache, method="unknown")
