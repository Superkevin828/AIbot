from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from local_english_model.cli import build_parser, main


class CLIParserTests(unittest.TestCase):
    def test_train_defaults(self) -> None:
        args = build_parser().parse_args(["train"])
        self.assertEqual(args.command, "train")
        self.assertEqual(args.max_steps, 2000)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.device, "auto")
        self.assertFalse(args.retrain_tokenizer)

    def test_generate_prompt(self) -> None:
        args = build_parser().parse_args(
            ["generate", "The", "sun", "--max-new-tokens", "12"]
        )
        self.assertEqual(args.prompt, ["The", "sun"])
        self.assertEqual(args.max_new_tokens, 12)

    def test_custom_paths(self) -> None:
        args = build_parser().parse_args(
            [
                "train",
                "--data-dir",
                "/tmp/texts",
                "--out-dir",
                "/tmp/checkpoints",
                "--n-layer",
                "2",
            ]
        )
        self.assertEqual(args.data_dir, Path("/tmp/texts"))
        self.assertEqual(args.out_dir, Path("/tmp/checkpoints"))
        self.assertEqual(args.n_layer, 2)

    def test_no_command_returns_usage(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main([])
        self.assertEqual(code, 2)
        self.assertIn("usage:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
