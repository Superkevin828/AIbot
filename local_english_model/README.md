# Local English Model

A small GPT-style language model that is trained from scratch on `.txt` files
stored on your computer. It uses PyTorch and a local BPE tokenizer. It does not
call an AI API, does not need an API key, and does not need an internet
connection at runtime.

This is a real neural Transformer, not an n-gram or retrieval script. It is also
not ChatGPT. It learns the patterns in whatever text you give it. A small model
trained on a small corpus will produce simple, repetitive text and cannot follow
instructions reliably.

## Requirements

- Python 3.10 or newer
- Enough disk space for PyTorch and your checkpoints
- A GPU is optional; CPU training works but is slower

Install the two required packages:

```bash
cd /root/AIbot/local_english_model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA build of PyTorch, follow the official PyTorch install selector instead
of installing the default package:

```text
https://pytorch.org/get-started/locally/
```

## Add training text

Put UTF-8 `.txt` files in:

```text
/root/AIbot/local_english_model/data/
```

Subfolders are read too. A starter file, `example_english.txt`, is included so
the project can be tested immediately. It is only 3.4 KB, which is far too small
for good results. Add books, articles, notes, or other text you have the right to
use. Several megabytes or more produce much better output than a few sentences.

## Train

```bash
cd /root/AIbot/local_english_model
python3 -m local_english_model train
```

The default model is deliberately small so it can run on a CPU:

```text
4 layers, 4 attention heads, 128 embedding width, 128-token context
```

The default run is 2,000 steps. For a quick smoke test:

```bash
python3 -m local_english_model train --max-steps 100 --eval-interval 25
```

For a larger model on a GPU:

```bash
python3 -m local_english_model train \
  --n-layer 6 --n-head 6 --n-embd 192 --block-size 256 \
  --batch-size 32 --grad-accum 2 --max-steps 5000
```

Training creates:

```text
checkpoints/
├── tokenizer.json   # BPE vocabulary learned from your text
├── latest.pt        # Most recent training state
└── best.pt          # Best validation loss seen so far
```

Continue an interrupted run:

```bash
python3 -m local_english_model train --resume checkpoints/latest.pt
```

The training loop uses AdamW, warmup plus cosine learning-rate decay, gradient
clipping, optional CUDA mixed precision, validation splits, and atomic checkpoint
saving. Add `--retrain-tokenizer` if you changed the corpus and want a fresh BPE
vocabulary.

## Generate text

```bash
python3 -m local_english_model generate "The sun rises in the"
python3 -m local_english_model generate "Learning a language" --max-new-tokens 120
```

Useful sampling options:

```text
--temperature          0 means greedy decoding; higher values are more random
--top-k                sample only from the most likely k tokens
--top-p                nucleus sampling threshold
--repetition-penalty   reduce repeated tokens
--seed                 make a run repeatable
--checkpoint           use a checkpoint other than checkpoints/latest.pt
--device               auto, cpu, cuda, or mps
```

## Interactive continuation

```bash
python3 -m local_english_model chat
```

This is text continuation, not an instruction-following assistant. If you type a
sentence, the model tries to continue that sentence. It does not have a separate
conversation memory or a hidden instruction-tuning stage.

## Inspect a checkpoint

```bash
python3 -m local_english_model info
```

This prints the architecture, parameter count, token counts, training step, and
validation loss.

## How it works

- Your `.txt` files are read as UTF-8 and split into documents.
- A ByteLevel BPE tokenizer is trained on the corpus and saved to disk.
- Each document is wrapped with beginning-of-sequence and end-of-sequence tokens.
- A decoder-only Transformer predicts the next token using causal self-attention.
- The model uses token and position embeddings, pre-LayerNorm blocks, GELU MLPs,
  dropout, and tied input/output embeddings.
- Training minimizes cross-entropy next-token loss on random context windows.
- Generation samples one token at a time using temperature, top-k, top-p, and a
  repetition penalty.

## Project layout

```text
local_english_model/
├── data/
│   └── example_english.txt
├── local_english_model/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── generate.py
│   ├── tokenizer.py
│   ├── train.py
│   └── transformer.py
├── tests/
├── requirements.txt
└── README.md
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Tests that require PyTorch or the tokenizer are skipped automatically when those
packages are not installed. A syntax-only check works without dependencies:

```bash
python3 -m compileall -q .
```

## Limitations

- It is a small model trained from scratch, not a pretrained large language model.
- It has no general knowledge beyond its training files.
- It cannot reliably follow instructions, reason, solve math, or hold a true chat.
- It has no safety alignment or moderation layer.
- Tiny corpora lead to memorization, repetition, and poor grammar.
- Training from scratch needs much more text and compute to approach a useful
  general-purpose assistant.
- Checkpoints contain Python metadata and are loaded with `torch.load`. Only load
  checkpoint files that you created or trust.

For stronger local conversational ability, use a pretrained local model such as a
llama.cpp-compatible model. That requires a large model download and additional
software, which this project deliberately avoids.
