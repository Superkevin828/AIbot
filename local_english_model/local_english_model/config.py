from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class ConfigError(RuntimeError):
    """Raised when a model or training configuration is invalid."""


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ConfigError("vocab_size must be at least 2")
        if self.block_size < 8:
            raise ConfigError("block_size must be at least 8")
        if self.n_layer < 1:
            raise ConfigError("n_layer must be at least 1")
        if self.n_head < 1:
            raise ConfigError("n_head must be at least 1")
        if self.n_embd < self.n_head:
            raise ConfigError("n_embd must be at least n_head")
        if self.n_embd % self.n_head != 0:
            raise ConfigError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("dropout must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GPTConfig":
        try:
            return cls(
                vocab_size=int(payload["vocab_size"]),
                block_size=int(payload.get("block_size", 128)),
                n_layer=int(payload.get("n_layer", 4)),
                n_head=int(payload.get("n_head", 4)),
                n_embd=int(payload.get("n_embd", 128)),
                dropout=float(payload.get("dropout", 0.1)),
                bias=bool(payload.get("bias", True)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError("Invalid model configuration in checkpoint") from exc
