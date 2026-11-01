from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "checkpoints"
_DEFAULT_CHECKPOINT = _DEFAULT_OUT_DIR / "latest.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-english-model",
        description=(
            "Train a GPT-style Transformer language model from scratch on local "
            ".txt files. No API keys and no external AI service."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser(
        "train",
        help="Train the Transformer from your text files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train.add_argument("--data-dir", type=Path, help="Folder containing .txt files")
    train.add_argument("--out-dir", type=Path, help="Checkpoint output folder")
    train.add_argument("--tokenizer", type=Path, help="BPE tokenizer JSON path")
    train.add_argument("--tokenizer-vocab-size", type=int, default=1024)
    train.add_argument(
        "--retrain-tokenizer",
        action="store_true",
        help="Ignore an existing tokenizer and train a new one",
    )
    train.add_argument("--block-size", type=int, default=128)
    train.add_argument("--n-layer", type=int, default=4)
    train.add_argument("--n-head", type=int, default=4)
    train.add_argument("--n-embd", type=int, default=128)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--grad-accum", type=int, default=1)
    train.add_argument("--max-steps", type=int, default=2000)
    train.add_argument("--eval-interval", type=int, default=100)
    train.add_argument("--eval-iters", type=int, default=20)
    train.add_argument("--log-interval", type=int, default=10)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--min-lr", type=float, default=3e-5)
    train.add_argument("--warmup-steps", type=int, default=100)
    train.add_argument("--weight-decay", type=float, default=0.1)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--val-fraction", type=float, default=0.05)
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    train.add_argument("--resume", type=Path, help="Continue from a checkpoint")
    train.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision",
    )
    train.set_defaults(handler=_handle_train)

    generate = subparsers.add_parser(
        "generate",
        help="Generate text from a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    generate.add_argument("prompt", nargs="*", help="Prompt text to continue")
    generate.add_argument("--checkpoint", type=Path, help="Checkpoint to load")
    generate.add_argument("--max-new-tokens", type=int, default=200)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=50)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--repetition-penalty", type=float, default=1.1)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--device", default="auto")
    generate.set_defaults(handler=_handle_generate)

    chat = subparsers.add_parser(
        "chat",
        help="Interactive text continuation (not an instruction-tuned assistant).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    chat.add_argument("--checkpoint", type=Path, help="Checkpoint to load")
    chat.add_argument("--max-new-tokens", type=int, default=120)
    chat.add_argument("--temperature", type=float, default=0.8)
    chat.add_argument("--top-k", type=int, default=50)
    chat.add_argument("--top-p", type=float, default=0.95)
    chat.add_argument("--repetition-penalty", type=float, default=1.1)
    chat.add_argument("--seed", type=int)
    chat.add_argument("--device", default="auto")
    chat.set_defaults(handler=_handle_chat)

    info = subparsers.add_parser(
        "info",
        help="Show checkpoint and model information.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    info.add_argument("--checkpoint", type=Path, help="Checkpoint to inspect")
    info.add_argument("--device", default="cpu")
    info.set_defaults(handler=_handle_info)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except ModuleNotFoundError as exc:
        print(
            f"Error: missing dependency '{exc.name}'. Install the tools with:\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1
    except (RuntimeError, OSError, ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _handle_train(args: argparse.Namespace) -> int:
    from .train import TrainSettings, train_model

    out_dir = _path(args.out_dir, _DEFAULT_OUT_DIR)
    tokenizer_path = _path(args.tokenizer, out_dir / "tokenizer.json")
    settings = TrainSettings(
        data_dir=_path(args.data_dir, _DEFAULT_DATA_DIR),
        out_dir=out_dir,
        tokenizer_path=tokenizer_path,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        retrain_tokenizer=args.retrain_tokenizer,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        use_amp=not args.no_amp,
    )
    train_model(settings)
    return 0


def _handle_generate(args: argparse.Namespace) -> int:
    from .generate import generate_text

    prompt = " ".join(args.prompt).strip()
    text = generate_text(
        _path(args.checkpoint, _DEFAULT_CHECKPOINT),
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        device=args.device,
    )
    print(text)
    return 0


def _handle_chat(args: argparse.Namespace) -> int:
    import torch

    from .train import load_for_inference, resolve_device

    device = resolve_device(args.device)
    model, tokenizer, _ = load_for_inference(
        _path(args.checkpoint, _DEFAULT_CHECKPOINT),
        device=device,
    )
    if args.seed is not None:
        torch.manual_seed(args.seed)

    print("GPT-style text continuation. This model is not instruction-tuned.")
    print("Type text to continue, /help for commands, or /quit to exit.")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line.lower() in {"/quit", "/exit", "quit", "exit"}:
            break
        if line == "/help":
            print("Type a prompt to continue it. Commands: /help, /quit")
            continue

        ids = [tokenizer.bos_id, *tokenizer.encode(line)]
        tensor = torch.tensor([ids], dtype=torch.long, device=device)
        output = model.generate(
            tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_id=tokenizer.eos_id,
        )
        print("model> " + tokenizer.decode(output[0].tolist()).strip())
    return 0


def _handle_info(args: argparse.Namespace) -> int:
    from .train import load_for_inference, resolve_device

    device = resolve_device(args.device)
    model, tokenizer, checkpoint = load_for_inference(
        _path(args.checkpoint, _DEFAULT_CHECKPOINT),
        device=device,
    )
    config = model.config
    print(f"Checkpoint: {_path(args.checkpoint, _DEFAULT_CHECKPOINT)}")
    print(f"Format: {checkpoint.get('format', 'unknown')}")
    print(f"Step: {int(checkpoint.get('step', 0)):,}")
    print(f"Best validation loss: {float(checkpoint.get('best_val_loss', 0)):.4f}")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Tokenizer vocabulary: {tokenizer.vocab_size:,}")
    print(
        f"Architecture: layers={config.n_layer} heads={config.n_head} "
        f"width={config.n_embd} block={config.block_size}"
    )
    tokens = checkpoint.get("tokens", {})
    if isinstance(tokens, dict):
        print(
            f"Tokens: train={int(tokens.get('train', 0)):,} "
            f"validation={int(tokens.get('validation', 0)):,}"
        )
    return 0


def _path(value: Path | None, default: Path) -> Path:
    return value.expanduser() if value is not None else default


if __name__ == "__main__":
    raise SystemExit(main())
