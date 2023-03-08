import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from preference_trainer.data import PreferencePair
from preference_trainer.model import CharTokenizer, TinyPreferencePolicy, sequence_logps


def response_key(prompt: str, completion: str) -> str:
    payload = json.dumps([prompt, completion], ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def model_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class ReferenceLogprobCache:
    entries: dict[str, float]
    reference_fingerprint: str

    def get(self, prompt: str, completion: str) -> torch.Tensor:
        key = response_key(prompt, completion)
        if key not in self.entries:
            raise KeyError("reference logprob is absent; rebuild the cache for this response")
        return torch.tensor(self.entries[key])

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "reference_fingerprint": self.reference_fingerprint,
            "entries": self.entries,
        }
        torch.save(payload, destination)

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceLogprobCache":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("version") != 1 or not isinstance(payload.get("entries"), dict):
            raise ValueError("unsupported or malformed reference cache")
        return cls(payload["entries"], payload["reference_fingerprint"])


def cached_responses(pairs: list[PreferencePair]) -> list[tuple[str, str]]:
    responses = {
        (pair.prompt, response)
        for pair in pairs
        for response in (pair.chosen, pair.rejected)
    }
    return sorted(responses)


@torch.no_grad()
def build_reference_cache(
    model: TinyPreferencePolicy,
    tokenizer: CharTokenizer,
    pairs: list[PreferencePair],
) -> ReferenceLogprobCache:
    responses = cached_responses(pairs)
    logps = sequence_logps(
        model,
        tokenizer,
        [prompt for prompt, _ in responses],
        [completion for _, completion in responses],
    )
    entries = {
        response_key(prompt, completion): float(logp)
        for (prompt, completion), logp in zip(responses, logps, strict=True)
    }
    return ReferenceLogprobCache(entries, model_fingerprint(model))


def reference_token_count(tokenizer: CharTokenizer, pairs: list[PreferencePair]) -> int:
    """Count full forward tokens for unique cached prompt-response sequences."""
    return sum(
        len(tokenizer.encode(prompt + completion, add_special_tokens=True)) - 1
        for prompt, completion in cached_responses(pairs)
    )
