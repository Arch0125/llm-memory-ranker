# Memory-Aware OpenAI Benchmark Sandbox

This repository is an experimentation workspace for:

- a typed memory layer with retrieval, gating, reranking, and prompt assembly
- OpenAI API benchmarking on LongMemEval

The core memory system leaves the generator unchanged. It stores hybrid benchmark memories (facts, episode summaries, and timeline memories), extracts temporal metadata, bundles relevant evidence, and then sends a structured evidence table to OpenAI for generation.

## Layout

- `memory/`: memory store, embeddings, retrieval, critic, policies, explainability
- `prompt/`: prompt assembly and budget selection
- `memory_cli.py`: inspect and manage local memories
- `benchmark_longmemeval_openai.py`: run the same benchmark with OpenAI as the generator
- `benchmark_longmemeval_retrieval.py`: emit an official-style LongMemEval retrieval log for the local memory retriever
- `run_longmemeval_protocol.py`: run a comparable LongMemEval protocol across `S full-history`, `S + memory`, and `Oracle upper bound`

## Quick Start

Install the Python dependencies you need inside the repo virtualenv:

```sh
./venv/bin/pip install httpx
```

Optional: install a stronger local retrieval encoder. The benchmark runner defaults to `benchmark-auto`, which uses `all-MiniLM-L6-v2` if `sentence-transformers` is installed and otherwise falls back to a temporal hashing embedder.

```sh
./venv/bin/pip install sentence-transformers
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
  --openai_model=gpt-4.1-mini \
  --output_path=reports/longmemeval_openai_predictions.jsonl \
  --details_path=reports/longmemeval_openai_details.jsonl \
  --summary_path=reports/longmemeval_openai_summary.json \
  --memory_enabled=True
```

Current defaults are tuned for LongMemEval temporal questions:

- `history_granularity=hybrid`
- `memory_embedder=benchmark-auto`
- session, fact, and timeline memories are all ingested
- retrieved memories are bundled into a compact evidence table before generation

`output_path` is intentionally compatible with the official LongMemEval evaluation script: each line contains only `question_id` and `hypothesis`. The companion `details_path` file includes local metrics such as exact match, token F1, selected memory count, selected-session recall, answerability, and OpenAI token usage.

To benchmark the hosted model without the memory layer, rerun with `--memory_enabled=False` and compare the two summary JSON files.

## Comparable LongMemEval Protocol

For a more faithful LongMemEval-style comparison, do not compare the memory layer against a question-only baseline. Instead, compare:

- `S full-history baseline`: the reader gets the full `longmemeval_s_cleaned.json` chat history directly
- `S + memory`: the same reader gets only the retrieved evidence selected by the local memory layer
- `Oracle upper bound`: the reader gets only the oracle evidence sessions from `longmemeval_oracle.json`

Download the standard data files first:

```sh
mkdir -p data
curl -L https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json -o data/longmemeval_s_cleaned.json
curl -L https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json -o data/longmemeval_oracle.json
```

Then run the protocol:

```sh
export OPENAI_API_KEY=...

./venv/bin/python run_longmemeval_protocol.py \
  --openai_model=gpt-4.1-mini \
  --max_examples=100
```

This will write:

- `reports/longmemeval_protocol/s_full_history_predictions.jsonl`
- `reports/longmemeval_protocol/s_memory_predictions.jsonl`
- `reports/longmemeval_protocol/oracle_upper_bound_predictions.jsonl`
- `reports/longmemeval_protocol/s_memory_retrievallog_session_memory.jsonl`
- `reports/longmemeval_protocol/s_memory_retrievallog_turn_memory.jsonl`
- matching `details.jsonl` and `summary.json` files for each condition
- `reports/longmemeval_protocol/manifest.json`

If you already have a local checkout of the official LongMemEval repo, you can also run the official evaluator on the generated `jsonl` files:

```sh
./venv/bin/python run_longmemeval_protocol.py \
  --openai_model=gpt-4.1-mini \
  --max_examples=100 \
  --official_repo_path=/path/to/LongMemEval
```

When `--official_repo_path` is set, the script looks for `evaluate_qa.py` and `print_qa_metrics.py` under that repo and writes the official evaluator output into `reports/longmemeval_protocol/*_official_*.txt`.

It also runs the official retrieval metric printer `print_retrieval_metrics.py` on the generated retrieval logs, so you get both:

- official QA evaluation on `jsonl` predictions
- official retrieval evaluation on `retrievallog_*` files

You can still use `benchmark_longmemeval_openai.py` directly for one-off conditions. The reader mode is controlled by `--reader_context_mode`:

- `memory`: local memory layer retrieval + evidence table
- `full-history`: feed the full chat history directly to the model
- `oracle-history`: feed only the answer sessions directly to the model
- `question-only`: question without history

If you only want the official-style retrieval log without the QA runs:

```sh
./venv/bin/python benchmark_longmemeval_retrieval.py \
  --dataset_path=data/longmemeval_s_cleaned.json \
  --granularity=turn \
  --max_examples=100 \
  --output_path=reports/longmemeval_retrievallog_turn_memory.jsonl \
  --summary_path=reports/longmemeval_retrievallog_turn_memory_summary.json
```

The resulting `retrievallog_*` file follows the same schema used by the official LongMemEval retrieval pipeline and can be passed to the official `print_retrieval_metrics.py`.

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
