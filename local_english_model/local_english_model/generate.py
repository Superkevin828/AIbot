from __future__ import annotations

from pathlib import Path

import torch

from .train import load_for_inference, resolve_device


class GenerationError(RuntimeError):
    """Raised when text generation cannot run."""


def generate_text(
    checkpoint_path: str | Path,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    seed: int | None = None,
    device: str = "auto",
) -> str:
    if max_new_tokens < 1:
        raise GenerationError("max_new_tokens must be at least 1")
    if temperature < 0:
        raise GenerationError("temperature cannot be negative")

    resolved_device = resolve_device(device)
    model, tokenizer, _ = load_for_inference(
        checkpoint_path,
        device=resolved_device,
    )

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    prompt_ids = tokenizer.encode(prompt)
    token_ids = [tokenizer.bos_id, *prompt_ids]
    tensor = torch.tensor([token_ids], dtype=torch.long, device=resolved_device)

    output = model.generate(
        tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_id=tokenizer.eos_id,
    )
    return tokenizer.decode(output[0].tolist()).strip()
