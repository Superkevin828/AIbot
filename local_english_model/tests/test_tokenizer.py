from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from local_english_model.tokenizer import BPETokenizer, TokenizerError

HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None


@unittest.skipUnless(HAS_TOKENIZERS, "tokenizers is not installed")
class BPETokenizerTests(unittest.TestCase):
    def test_train_round_trip_and_reload(self) -> None:
        texts = [
            "The sun rises in the east and sets in the west.",
            "Water turns into ice when the temperature is low.",
            "Learning a language takes regular practice.",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer = BPETokenizer.train(
                texts,
                vocab_size=320,
                save_path=path,
                min_frequency=1,
            )
            self.assertTrue(path.is_file())
            self.assertGreaterEqual(tokenizer.vocab_size, 260)
            self.assertNotEqual(tokenizer.bos_id, tokenizer.eos_id)
            self.assertNotEqual(tokenizer.pad_id, tokenizer.unk_id)

            text = "The sun rises in the east."
            ids = tokenizer.encode(text)
            self.assertTrue(ids)
            self.assertEqual(tokenizer.decode(ids), text)

            document_ids = tokenizer.document_ids(text)
            self.assertEqual(document_ids[0], tokenizer.bos_id)
            self.assertEqual(document_ids[-1], tokenizer.eos_id)

            loaded = BPETokenizer.load(path)
            self.assertEqual(loaded.vocab_size, tokenizer.vocab_size)
            self.assertEqual(loaded.encode(text), ids)
            self.assertEqual(loaded.decode(ids), text)

    def test_rejects_tiny_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TokenizerError):
                BPETokenizer.train(
                    ["hello"],
                    vocab_size=4,
                    save_path=Path(directory) / "tokenizer.json",
                )

    def test_rejects_empty_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TokenizerError):
                BPETokenizer.train(
                    [],
                    vocab_size=300,
                    save_path=Path(directory) / "tokenizer.json",
                )

    def test_missing_file(self) -> None:
        with self.assertRaises(TokenizerError):
            BPETokenizer.load("/tmp/does-not-exist-tokenizer.json")


if __name__ == "__main__":
    unittest.main()
