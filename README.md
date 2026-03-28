# Memory-Aware OpenAI Benchmark Sandbox

This repository is an experimentation workspace for:

- a typed memory layer with retrieval, gating, reranking, and prompt assembly
- OpenAI API benchmarking on LongMemEval

The core memory system leaves the generator unchanged. It retrieves candidate memories from a local SQLite store, scores whether they should be used, and then injects only the selected memories into the prompt before sending the request to OpenAI.

## Layout

- `memory/`: memory store, embeddings, retrieval, critic, policies, explainability
- `prompt/`: prompt assembly and budget selection
- `memory_cli.py`: inspect and manage local memories
- `benchmark_longmemeval_openai.py`: run the same benchmark with OpenAI as the generator

## Quick Start

Install the Python dependencies you need inside the repo virtualenv:

```sh
./venv/bin/pip install httpx
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

Run the same benchmark with OpenAI as the generator while keeping retrieval and gating local:

```sh
export OPENAI_API_KEY=...

./venv/bin/python benchmark_longmemeval_openai.py \
  --dataset_path=data/longmemeval_oracle.json \
  --max_examples=25 \
  --openai_model=gpt-5-mini \
  --output_path=reports/longmemeval_openai_predictions.jsonl \
  --details_path=reports/longmemeval_openai_details.jsonl \
  --summary_path=reports/longmemeval_openai_summary.json \
  --memory_enabled=True
```

`output_path` is intentionally compatible with the official LongMemEval evaluation script: each line contains only `question_id` and `hypothesis`. The companion `details_path` file includes local metrics such as exact match, token F1, selected memory count, selected-session recall, and OpenAI token usage.

To benchmark the hosted model without the memory layer, rerun with `--memory_enabled=False` and compare the two summary JSON files.

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

## Standard Benchmark Data

For a public long-term memory benchmark, use LongMemEval. Download one of the official JSON files into `data/`:

```sh
mkdir -p data
curl -L https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json -o data/longmemeval_oracle.json
```

For quick validation, `longmemeval_oracle.json` is the easiest starting point because it contains only the evidence sessions. For harder retrieval stress tests, use `longmemeval_s_cleaned.json` or `longmemeval_m_cleaned.json` with the same OpenAI runner.
