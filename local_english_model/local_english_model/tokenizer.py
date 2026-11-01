from __future__ import annotations

from pathlib import Path
from typing import Iterable


class TokenizerError(RuntimeError):
    """Raised when the BPE tokenizer cannot be trained or loaded."""


class BPETokenizer:
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        ids = {
            name: tokenizer.token_to_id(name)
            for name in self.SPECIAL_TOKENS
        }
        if any(value is None for value in ids.values()):
            raise TokenizerError("Tokenizer is missing required special tokens")
        self.pad_id = int(ids[self.PAD_TOKEN])
        self.bos_id = int(ids[self.BOS_TOKEN])
        self.eos_id = int(ids[self.EOS_TOKEN])
        self.unk_id = int(ids[self.UNK_TOKEN])

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int,
        save_path: str | Path,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        minimum_vocab_size = len(cls.SPECIAL_TOKENS) + 256
        if vocab_size < minimum_vocab_size:
            raise TokenizerError(
                f"vocab_size must be at least {minimum_vocab_size} for ByteLevel BPE"
            )

        try:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        except ImportError as exc:
            raise TokenizerError(
                "tokenizers is not installed. Run: pip install -r requirements.txt"
            ) from exc

        materials = list(texts)
        if not materials or not any(text.strip() for text in materials):
            raise TokenizerError("No text was provided for tokenizer training")

        tokenizer = Tokenizer(models.BPE(unk_token=cls.UNK_TOKEN))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(cls.SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(
            iter(materials),
            trainer=trainer,
            length=len(materials),
        )

        path = Path(save_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(path))
        return cls(tokenizer)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise TokenizerError(
                "tokenizers is not installed. Run: pip install -r requirements.txt"
            ) from exc

        source = Path(path).expanduser()
        if not source.is_file():
            raise TokenizerError(f"Tokenizer file not found: {source}")
        try:
            return cls(Tokenizer.from_file(str(source)))
        except Exception as exc:  # noqa: BLE001 - normalize tokenizer failures
            raise TokenizerError(f"Could not load tokenizer {source}: {exc}") from exc

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())

    def encode(self, text: str) -> list[int]:
        return list(
            self._tokenizer.encode(text, add_special_tokens=False).ids
        )

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def document_ids(self, text: str) -> list[int]:
        return [self.bos_id, *self.encode(text), self.eos_id]
