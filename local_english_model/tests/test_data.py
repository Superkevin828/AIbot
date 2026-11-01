from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from local_english_model.data import (
        DataError,
        TextDocument,
        build_token_data,
        load_texts,
    )


class FakeTokenizer:
    bos_id = 1
    eos_id = 2

    def document_ids(self, text: str) -> list[int]:
        body = [3 + (ord(character) % 20) for character in text]
        return [self.bos_id, *body, self.eos_id]


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class DataTests(unittest.TestCase):
    def test_load_texts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("First text.", encoding="utf-8")
            (root / "empty.txt").write_text("   \n", encoding="utf-8")
            (root / ".hidden.txt").write_text("Hidden text.", encoding="utf-8")
            (root / "notes.md").write_text("Not training text.", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "b.txt").write_text(
                "Second text.", encoding="utf-8"
            )

            documents = load_texts(root)
            self.assertEqual(
                [document.source for document in documents],
                ["a.txt", "nested/b.txt"],
            )
            self.assertEqual(documents[0].text, "First text.")

    def test_load_texts_requires_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(DataError):
                load_texts(directory)

    def test_hidden_parent_directory_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".hidden"
            root.mkdir()
            (root / "text.txt").write_text("Visible text.", encoding="utf-8")
            documents = load_texts(root)
            self.assertEqual([document.source for document in documents], ["text.txt"])

    def test_build_token_data_and_batches(self) -> None:
        documents = [
            TextDocument(
                source="sample.txt",
                text="The sun rises in the east. " * 8,
            )
        ]
        data = build_token_data(
            documents,
            FakeTokenizer(),
            block_size=8,
            val_fraction=0.2,
        )
        self.assertEqual(data.document_count, 1)
        self.assertEqual(
            data.total_tokens,
            len(data.train_data) + len(data.val_data),
        )
        self.assertGreater(len(data.train_data), 8)
        self.assertGreater(len(data.val_data), 8)

        x, y = data.get_batch("train", batch_size=4, device=torch.device("cpu"))
        self.assertEqual(tuple(x.shape), (4, 8))
        self.assertEqual(tuple(y.shape), (4, 8))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))

    def test_rejects_short_text(self) -> None:
        documents = [TextDocument(source="tiny.txt", text="Too short.")]
        with self.assertRaises(DataError):
            build_token_data(
                documents,
                FakeTokenizer(),
                block_size=64,
            )

    def test_rejects_invalid_options(self) -> None:
        documents = [TextDocument(source="tiny.txt", text="Text " * 40)]
        with self.assertRaises(DataError):
            build_token_data(
                documents,
                FakeTokenizer(),
                block_size=8,
                val_fraction=0.0,
            )
        data = build_token_data(
            documents,
            FakeTokenizer(),
            block_size=8,
            val_fraction=0.2,
        )
        with self.assertRaises(DataError):
            data.get_batch("test", batch_size=1, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
