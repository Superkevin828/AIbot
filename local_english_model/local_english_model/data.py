from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .tokenizer import BPETokenizer


class DataError(RuntimeError):
    """Raised when training text cannot be prepared."""


@dataclass(frozen=True)
class TextDocument:
    source: str
    text: str


@dataclass
class TokenData:
    train_data: torch.Tensor
    val_data: torch.Tensor
    block_size: int
    document_count: int
    total_tokens: int
    source_characters: int

    def get_batch(
        self,
        split: str,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size < 1:
            raise DataError("batch_size must be at least 1")
        if split == "train":
            data = self.train_data
        elif split == "val":
            data = self.val_data
        else:
            raise DataError(f"Unknown split: {split}")

        max_start = len(data) - self.block_size
        if max_start < 1:
            raise DataError(
                f"The {split} split is too short for block_size={self.block_size}"
            )

        starts = torch.randint(max_start, (batch_size,))
        x = torch.stack([data[index : index + self.block_size] for index in starts])
        y = torch.stack(
            [data[index + 1 : index + 1 + self.block_size] for index in starts]
        )
        return x.to(device), y.to(device)


def load_texts(data_dir: str | Path) -> list[TextDocument]:
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise DataError(f"Text folder not found: {root}")

    paths = sorted(
        path
        for path in root.rglob("*.txt")
        if path.is_file()
        and not any(
            part.startswith(".") for part in path.relative_to(root).parts
        )
    )
    if not paths:
        raise DataError(f"No .txt files found in {root}")

    documents: list[TextDocument] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            raise DataError(f"Could not read {path}: {exc}") from exc
        if not text.strip():
            continue
        documents.append(
            TextDocument(
                source=path.relative_to(root).as_posix(),
                text=text,
            )
        )

    if not documents:
        raise DataError(f"No usable text found in {root}")
    return documents


def build_token_data(
    documents: list[TextDocument],
    tokenizer: BPETokenizer,
    *,
    block_size: int,
    val_fraction: float = 0.05,
) -> TokenData:
    if not 0.0 < val_fraction < 0.5:
        raise DataError("val_fraction must be between 0 and 0.5")
    if block_size < 8:
        raise DataError("block_size must be at least 8")

    ids: list[int] = []
    for document in documents:
        ids.extend(tokenizer.document_ids(document.text))

    total = len(ids)
    minimum = (block_size + 2) * 2
    if total < minimum:
        raise DataError(
            f"Only {total:,} tokens were produced, but at least {minimum:,} are "
            "needed. Add more .txt data, lower --block-size, or use a smaller "
            "tokenizer vocabulary."
        )

    val_size = max(int(total * val_fraction), block_size + 2)
    train_size = total - val_size
    if train_size < block_size + 2:
        val_size = max(block_size + 2, total // 5)
        train_size = total - val_size
    if train_size < block_size + 2:
        raise DataError(
            "The text is too small to make both training and validation splits. "
            "Add more text or lower --block-size."
        )

    tokens = torch.tensor(ids, dtype=torch.long)
    return TokenData(
        train_data=tokens[:train_size],
        val_data=tokens[train_size:],
        block_size=block_size,
        document_count=len(documents),
        total_tokens=total,
        source_characters=sum(len(document.text) for document in documents),
    )
