import torch

from preference_trainer.cache import (
    ReferenceLogprobCache,
    build_reference_cache,
    reference_token_count,
    response_key,
)
from preference_trainer.data import preference_datasets
from preference_trainer.model import (
    CharTokenizer,
    TinyPolicyConfig,
    TinyPreferencePolicy,
    cached_forward_flops,
    matrix_forward_flops_per_token,
    sequence_logps,
)


def make_model() -> tuple[CharTokenizer, TinyPreferencePolicy]:
    torch.manual_seed(7)
    tokenizer = CharTokenizer()
    model = TinyPreferencePolicy(TinyPolicyConfig(tokenizer.vocab_size))
    return tokenizer, model


def test_sequence_logps_match_independent_teacher_forcing() -> None:
    tokenizer, model = make_model()
    prompt = "Q: Two plus two?\nA:"
    completion = " Four."
    actual = sequence_logps(model, tokenizer, [prompt], [completion])

    ids = torch.tensor(tokenizer.encode(prompt + completion, add_special_tokens=True)).unsqueeze(0)
    logits = model(ids[:, :-1])[0]
    prompt_length = 1 + len(tokenizer.encode(prompt))
    target_ids = ids[0, 1:]
    expected_terms = []
    for index, target in enumerate(target_ids):
        if index + 1 >= prompt_length:
            expected_terms.append(torch.log_softmax(logits[index], dim=-1)[target])
    expected = torch.stack(expected_terms).sum()
    assert torch.allclose(actual[0], expected)


def test_cache_is_content_addressed_and_survives_roundtrip(tmp_path) -> None:
    tokenizer, model = make_model()
    train, _ = preference_datasets()
    pairs = train[:2]
    cache = build_reference_cache(model, tokenizer, pairs)
    pair = pairs[0]
    expected = sequence_logps(model, tokenizer, [pair.prompt], [pair.chosen])[0]

    assert torch.allclose(cache.get(pair.prompt, pair.chosen), expected)
    path = tmp_path / "reference.pt"
    cache.save(path)
    loaded = ReferenceLogprobCache.load(path)
    assert loaded.reference_fingerprint == cache.reference_fingerprint
    assert loaded.entries == cache.entries
    assert response_key(pair.prompt, pair.chosen) != response_key(pair.prompt, pair.rejected)

    with torch.no_grad():
        model.output.weight.add_(1.0)
    assert torch.allclose(loaded.get(pair.prompt, pair.chosen), expected)


def test_cache_rejects_unknown_response() -> None:
    cache = ReferenceLogprobCache({}, "fingerprint")
    try:
        cache.get("known", "unknown")
    except KeyError as error:
        assert "rebuild" in str(error)
    else:
        raise AssertionError("cache miss should fail rather than silently recompute")


def test_deterministic_disjoint_datasets() -> None:
    train_a, eval_a = preference_datasets()
    train_b, eval_b = preference_datasets()
    assert train_a == train_b
    assert eval_a == eval_b
    assert set(train_a).isdisjoint(eval_a)
    assert len(train_a) == 24
    assert len(eval_a) == 8


def test_analytical_cache_flops_use_full_sequence_tokens_and_epochs() -> None:
    tokenizer, model = make_model()
    train, _ = preference_datasets()
    tokens = reference_token_count(tokenizer, train[:2])
    per_token = matrix_forward_flops_per_token(model.config)
    expected_per_token = 6 * 24 * (16 + 24) + 2 * 24 * tokenizer.vocab_size
    assert per_token == expected_per_token
    assert cached_forward_flops(model.config, tokens, epochs=3) == 3 * tokens * expected_per_token
