from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence


class CharTokenizer:
    """A fixed printable-ASCII tokenizer with explicit BOS and EOS tokens."""

    alphabet = "\n" + "".join(chr(code) for code in range(32, 127))
    bos_id = 0
    eos_id = 1

    def __init__(self) -> None:
        self._to_id = {character: index + 2 for index, character in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet) + 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        unknown = [character for character in text if character not in self._to_id]
        if unknown:
            raise ValueError(f"characters outside fixed vocabulary: {unknown!r}")
        ids = [self._to_id[character] for character in text]
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids


@dataclass(frozen=True)
class TinyPolicyConfig:
    vocab_size: int
    embedding_dim: int = 16
    hidden_dim: int = 24


class TinyPreferencePolicy(nn.Module):
    """A trainable character-level GRU policy, not a pretrained language model."""

    def __init__(self, config: TinyPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, config.hidden_dim, batch_first=True)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        hidden, _ = self.gru(self.embedding(token_ids))
        return self.output(hidden)


def sequence_logps(
    model: TinyPreferencePolicy,
    tokenizer: CharTokenizer,
    prompts: list[str],
    completions: list[str],
) -> Tensor:
    """Teacher-force completion tokens and sum their conditional logprobs."""
    if len(prompts) != len(completions) or not prompts:
        raise ValueError("prompts and completions must be non-empty and equally sized")
    sequences = [
        torch.tensor(tokenizer.encode(prompt + completion, add_special_tokens=True))
        for prompt, completion in zip(prompts, completions, strict=True)
    ]
    prompt_lengths = [1 + len(tokenizer.encode(prompt)) for prompt in prompts]
    batch = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.bos_id)
    logits = model(batch[:, :-1])
    targets = batch[:, 1:]
    token_logps = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    positions = torch.arange(1, batch.shape[1]).unsqueeze(0)
    lengths = torch.tensor([len(sequence) for sequence in sequences]).unsqueeze(1)
    completion_starts = torch.tensor(prompt_lengths).unsqueeze(1)
    completion_mask = (positions >= completion_starts) & (positions < lengths)
    return (token_logps * completion_mask).sum(dim=1)


def matrix_forward_flops_per_token(config: TinyPolicyConfig) -> int:
    """Count GRU and output matrix multiply-adds as two FLOPs each."""
    hidden = config.hidden_dim
    gru_flops = 6 * hidden * (config.embedding_dim + hidden)
    output_flops = 2 * hidden * config.vocab_size
    return gru_flops + output_flops


def cached_forward_flops(config: TinyPolicyConfig, token_count: int, epochs: int = 1) -> int:
    """Analytical matrix FLOPs avoided by reading rather than recomputing a cache."""
    if token_count < 0 or epochs < 0:
        raise ValueError("token_count and epochs must be nonnegative")
    return matrix_forward_flops_per_token(config) * token_count * epochs
