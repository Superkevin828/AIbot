from __future__ import annotations

import unittest

from local_english_model.config import ConfigError, GPTConfig


class GPTConfigTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        config = GPTConfig(
            vocab_size=512,
            block_size=64,
            n_layer=2,
            n_head=4,
            n_embd=32,
            dropout=0.05,
        )
        restored = GPTConfig.from_dict(config.to_dict())
        self.assertEqual(config, restored)

    def test_rejects_invalid_values(self) -> None:
        with self.assertRaises(ConfigError):
            GPTConfig(vocab_size=1)
        with self.assertRaises(ConfigError):
            GPTConfig(vocab_size=10, block_size=4)
        with self.assertRaises(ConfigError):
            GPTConfig(vocab_size=10, n_layer=0)
        with self.assertRaises(ConfigError):
            GPTConfig(vocab_size=10, n_head=0)
        with self.assertRaises(ConfigError):
            GPTConfig(vocab_size=10, n_head=3, n_embd=8)
        with self.assertRaises(ConfigError):
            GPTConfig(vocab_size=10, dropout=1.0)

    def test_rejects_missing_vocab_size(self) -> None:
        with self.assertRaises(ConfigError):
            GPTConfig.from_dict({})


if __name__ == "__main__":
    unittest.main()
