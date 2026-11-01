from __future__ import annotations

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from local_english_model.train import _learning_rate


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class LearningRateTests(unittest.TestCase):
    def test_warmup(self) -> None:
        first = _learning_rate(
            0,
            max_steps=100,
            warmup_steps=10,
            learning_rate=1e-3,
            min_lr=1e-4,
        )
        last = _learning_rate(
            9,
            max_steps=100,
            warmup_steps=10,
            learning_rate=1e-3,
            min_lr=1e-4,
        )
        self.assertAlmostEqual(first, 1e-4)
        self.assertAlmostEqual(last, 1e-3)

    def test_cosine_decay_ends_at_minimum(self) -> None:
        start = _learning_rate(
            10,
            max_steps=100,
            warmup_steps=10,
            learning_rate=1e-3,
            min_lr=1e-4,
        )
        end = _learning_rate(
            100,
            max_steps=100,
            warmup_steps=10,
            learning_rate=1e-3,
            min_lr=1e-4,
        )
        self.assertAlmostEqual(start, 1e-3)
        self.assertAlmostEqual(end, 1e-4)

    def test_short_schedule(self) -> None:
        value = _learning_rate(
            4,
            max_steps=5,
            warmup_steps=10,
            learning_rate=1e-3,
            min_lr=1e-4,
        )
        self.assertAlmostEqual(value, 5e-4)


if __name__ == "__main__":
    unittest.main()
