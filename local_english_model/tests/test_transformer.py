from __future__ import annotations

import importlib.util
import unittest

from local_english_model.config import GPTConfig

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from local_english_model.transformer import GPT


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class GPTTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = GPTConfig(
            vocab_size=48,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
        self.model = GPT(self.config)
        self.model.eval()

    def test_forward_loss_and_shapes(self) -> None:
        x = torch.randint(0, self.config.vocab_size, (3, 8))
        y = torch.randint(0, self.config.vocab_size, (3, 8))
        logits, loss = self.model(x, y)
        self.assertEqual(tuple(logits.shape), (3, 8, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(self.model.parameter_count(), 0)

        last_logits, no_loss = self.model(x)
        self.assertEqual(tuple(last_logits.shape), (3, 1, self.config.vocab_size))
        self.assertIsNone(no_loss)

    def test_causality(self) -> None:
        x = torch.randint(0, self.config.vocab_size, (1, 12))
        changed = x.clone()
        changed[0, -1] = (int(changed[0, -1]) + 1) % self.config.vocab_size
        targets = torch.randint(0, self.config.vocab_size, (1, 12))

        first, _ = self.model(x, targets)
        second, _ = self.model(changed, targets)
        self.assertTrue(torch.allclose(first[:, :-1], second[:, :-1]))

    def test_generate(self) -> None:
        prompt = torch.randint(0, self.config.vocab_size, (1, 4))
        output = self.model.generate(
            prompt,
            max_new_tokens=5,
            temperature=0,
        )
        self.assertEqual(tuple(output.shape), (1, 9))
        self.assertTrue(torch.equal(output[:, :4], prompt))

        with self.assertRaises(ValueError):
            self.model.generate(prompt, max_new_tokens=1, top_p=0)
        with self.assertRaises(ValueError):
            self.model.generate(prompt, max_new_tokens=1, top_k=-1)
        with self.assertRaises(ValueError):
            self.model.generate(prompt, max_new_tokens=1, repetition_penalty=0)

    def test_rejects_sequence_longer_than_block(self) -> None:
        x = torch.randint(0, self.config.vocab_size, (1, 17))
        with self.assertRaises(ValueError):
            self.model(x)


if __name__ == "__main__":
    unittest.main()
