# Memory-Aware Local Inference Sandbox

This repository is a local experimentation workspace for:

- small decoder-only language model training and sampling with PyTorch
- a typed memory layer with retrieval, gating, reranking, and prompt assembly
- Apple Silicon inference through `mlx-lm`

The core memory system leaves the base language model unchanged. It retrieves candidate memories from a local SQLite store, scores whether they should be used, and then injects only the selected memories into the prompt.

## Layout

- `train.py`, `model.py`, `sample.py`: baseline Torch training and sampling
- `memory/`: memory store, embeddings, retrieval, critic, policies, explainability
- `prompt/`: prompt assembly and budget selection
- `memory_cli.py`: inspect and manage local memories
- `sample_mlx.py`: Apple Silicon runner using `mlx-lm`

## Quick Start

Install the Python dependencies you need inside the repo virtualenv. For the Apple Silicon path:

```sh
./venv/bin/pip install mlx-lm
```

Seed a couple of memories:

```sh
./venv/bin/python memory_cli.py --db-path memory.sqlite add \
  --text="Working on the memory-aware inference project." \
  --type=project \
  --importance=0.9

./venv/bin/python memory_cli.py --db-path memory.sqlite add \
  --text="Prefers concise technical explanations." \
  --type=preference \
  --importance=0.8
```

Preview retrieval:

```sh
./venv/bin/python memory_cli.py --db-path memory.sqlite search \
  --query="Debug the memory retrieval bug in the inference project." \
  --show-prompt
```

Run the Apple Silicon path with the default MLX instruct model:

```sh
./venv/bin/python sample_mlx.py \
  --start="Debug the memory retrieval bug in the inference project." \
  --max_new_tokens=160 \
  --temperature=0.3 \
  --top_k=40 \
  --memory_enabled=True \
  --memory_db_path=memory.sqlite \
  --memory_user_id=default \
  --memory_explain=True
```

Run the Torch sampler with the local memory layer:

```sh
./venv/bin/python sample.py \
  --init_from=gpt2 \
  --start="Explain the memory retrieval bug." \
  --num_samples=1 \
  --max_new_tokens=120 \
  --temperature=0.2 \
  --top_k=40 \
  --device=cpu \
  --compile=False \
  --memory_enabled=True \
  --memory_db_path=memory.sqlite \
  --memory_user_id=default \
  --memory_explain=True
```

## Memory Layer

Each memory item stores:

- text
- type
- timestamps
- retrieval count
- importance
- status and optional version group
- metadata JSON
- embedding vector
- event history

Selection is weighted by similarity, type-aware critic confidence, importance, freshness, and prompt budget. The selected items are tagged in the prompt so the generator can treat them as contextual evidence instead of unconditional instructions.

## Verification

Run the tests with:

```sh
./venv/bin/python -m unittest discover -s tests
```
