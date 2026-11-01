from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import GPTConfig
from .data import TokenData, build_token_data, load_texts
from .tokenizer import BPETokenizer, TokenizerError
from .transformer import GPT


class TrainingError(RuntimeError):
    """Raised when model training cannot continue."""


@dataclass(frozen=True)
class TrainSettings:
    data_dir: Path
    out_dir: Path
    tokenizer_path: Path
    tokenizer_vocab_size: int = 1024
    retrain_tokenizer: bool = False
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    batch_size: int = 16
    grad_accum: int = 1
    max_steps: int = 2000
    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 10
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_fraction: float = 0.05
    seed: int = 1337
    device: str = "auto"
    resume: Path | None = None
    use_amp: bool = True


def train_model(settings: TrainSettings) -> Path:
    _validate_settings(settings)
    _set_seed(settings.seed)
    device = _resolve_device(settings.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    out_dir = settings.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    documents = load_texts(settings.data_dir)
    texts = [document.text for document in documents]

    resume_path = settings.resume.expanduser() if settings.resume else None
    checkpoint = None
    start_step = 0
    best_val_loss = float("inf")

    if resume_path is not None:
        if not resume_path.is_file():
            raise TrainingError(f"Checkpoint not found: {resume_path}")
        checkpoint = _load_checkpoint(resume_path, device)
        try:
            config = GPTConfig.from_dict(checkpoint["config"])
            tokenizer_path = Path(checkpoint["tokenizer_path"]).expanduser()
            tokenizer = BPETokenizer.load(tokenizer_path)
        except (KeyError, TypeError) as exc:
            raise TrainingError("Checkpoint is incomplete") from exc
        if tokenizer.vocab_size != config.vocab_size:
            raise TrainingError(
                "Tokenizer vocabulary does not match the checkpoint model"
            )
        start_step = int(checkpoint.get("step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"Resuming from {resume_path} at step {start_step:,}")
    else:
        tokenizer_path = settings.tokenizer_path.expanduser()
        if tokenizer_path.is_file() and not settings.retrain_tokenizer:
            tokenizer = BPETokenizer.load(tokenizer_path)
            print(f"Loaded tokenizer from {tokenizer_path}")
        else:
            print("Training BPE tokenizer...")
            tokenizer = BPETokenizer.train(
                texts,
                vocab_size=settings.tokenizer_vocab_size,
                save_path=tokenizer_path,
            )
            print(
                f"Tokenizer vocabulary: {tokenizer.vocab_size:,} "
                f"({tokenizer_path})"
            )
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=settings.block_size,
            n_layer=settings.n_layer,
            n_head=settings.n_head,
            n_embd=settings.n_embd,
            dropout=settings.dropout,
            bias=True,
        )

    data = build_token_data(
        documents,
        tokenizer,
        block_size=config.block_size,
        val_fraction=settings.val_fraction,
    )
    print(
        f"Data: {data.document_count} files, {data.total_tokens:,} tokens "
        f"({len(data.train_data):,} train / {len(data.val_data):,} validation)"
    )

    model = GPT(config).to(device)
    optimizer = _configure_optimizer(
        model,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    use_amp = settings.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    if checkpoint is not None:
        try:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
        except (KeyError, RuntimeError) as exc:
            raise TrainingError(f"Could not restore checkpoint: {exc}") from exc

    print(
        f"Model: {model.parameter_count():,} parameters | "
        f"layers={config.n_layer} heads={config.n_head} "
        f"width={config.n_embd} block={config.block_size}"
    )
    print(f"Device: {device} | AMP: {'on' if use_amp else 'off'}")

    if start_step >= settings.max_steps:
        print("Checkpoint is already at or beyond --max-steps.")
        return _save_checkpoint(
            out_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            tokenizer_path=tokenizer_path,
            step=start_step,
            best_val_loss=best_val_loss,
            data=data,
        )

    model.train()
    for step in range(start_step, settings.max_steps):
        learning_rate = _learning_rate(
            step,
            max_steps=settings.max_steps,
            warmup_steps=settings.warmup_steps,
            learning_rate=settings.learning_rate,
            min_lr=settings.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(settings.grad_accum):
            x, y = data.get_batch(
                "train",
                batch_size=settings.batch_size,
                device=device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _, loss = model(x, y)
            if loss is None:
                raise TrainingError("Model did not return a training loss")
            scaled_loss = loss / settings.grad_accum
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            accumulated_loss += scaled_loss.item()

        if settings.grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                settings.grad_clip,
            )
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step % settings.log_interval == 0:
            print(
                f"step {step:6d} | loss {accumulated_loss:.4f} | "
                f"lr {learning_rate:.2e}"
            )

        is_last = step == settings.max_steps - 1
        if step % settings.eval_interval == 0 or is_last:
            losses = estimate_loss(
                model,
                data,
                batch_size=settings.batch_size,
                eval_iters=settings.eval_iters,
                device=device,
                use_amp=use_amp,
            )
            print(
                f"eval  {step:6d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"ppl {math.exp(min(losses['val'], 20)):.2f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_path = _save_checkpoint(
                    out_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    tokenizer_path=tokenizer_path,
                    step=step + 1,
                    best_val_loss=best_val_loss,
                    data=data,
                )
                print(f"Saved new best checkpoint: {best_path}")
            latest_path = _save_checkpoint(
                out_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                tokenizer_path=tokenizer_path,
                step=step + 1,
                best_val_loss=best_val_loss,
                data=data,
            )
            model.train()

    final_path = _save_checkpoint(
        out_dir / "latest.pt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        tokenizer_path=tokenizer_path,
        step=settings.max_steps,
        best_val_loss=best_val_loss,
        data=data,
    )
    print(f"Training complete. Checkpoint: {final_path}")
    return final_path


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data: TokenData,
    *,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    if eval_iters < 1:
        raise TrainingError("eval_iters must be at least 1")
    model.eval()
    results: dict[str, float] = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            x, y = data.get_batch(split, batch_size=batch_size, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _, loss = model(x, y)
            if loss is None:
                raise TrainingError("Model did not return an evaluation loss")
            losses.append(float(loss.item()))
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


def load_for_inference(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[GPT, BPETokenizer, dict]:
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise TrainingError(f"Checkpoint not found: {path}")
    checkpoint = _load_checkpoint(path, device)
    try:
        config = GPTConfig.from_dict(checkpoint["config"])
        tokenizer = BPETokenizer.load(checkpoint["tokenizer_path"])
        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
    except (KeyError, TypeError, RuntimeError, TokenizerError) as exc:
        raise TrainingError(f"Could not load checkpoint: {exc}") from exc
    model.to(device)
    model.eval()
    return model, tokenizer, checkpoint


def resolve_device(name: str) -> torch.device:
    return _resolve_device(name)


def _validate_settings(settings: TrainSettings) -> None:
    if settings.batch_size < 1:
        raise TrainingError("batch_size must be at least 1")
    if settings.grad_accum < 1:
        raise TrainingError("grad_accum must be at least 1")
    if settings.max_steps < 1:
        raise TrainingError("max_steps must be at least 1")
    if settings.eval_interval < 1:
        raise TrainingError("eval_interval must be at least 1")
    if settings.log_interval < 1:
        raise TrainingError("log_interval must be at least 1")
    if settings.eval_iters < 1:
        raise TrainingError("eval_iters must be at least 1")
    if settings.learning_rate <= 0:
        raise TrainingError("learning_rate must be positive")
    if settings.warmup_steps < 0:
        raise TrainingError("warmup_steps cannot be negative")


def _configure_optimizer(
    model: GPT,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dim() >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
    )


def _learning_rate(
    step: int,
    *,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_lr: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    if max_steps <= warmup_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    ratio = min(max(ratio, 0.0), 1.0)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coefficient * (learning_rate - min_lr)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    value = name.strip().lower()
    if value in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise TrainingError("MPS was requested but is not available")
    return device


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(
            str(path),
            map_location=device,
            weights_only=False,
        )
    except Exception as exc:  # noqa: BLE001 - normalize checkpoint failures
        raise TrainingError(f"Could not load checkpoint {path}: {exc}") from exc


def _save_checkpoint(
    path: Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler,
    config: GPTConfig,
    tokenizer_path: Path,
    step: int,
    best_val_loss: float,
    data: TokenData,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "local-english-gpt-v1",
        "config": config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "tokens": {
            "train": len(data.train_data),
            "validation": len(data.val_data),
            "total": data.total_tokens,
        },
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception as exc:  # noqa: BLE001 - normalize checkpoint failures
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise TrainingError(f"Could not save checkpoint to {path}: {exc}") from exc
    return path
