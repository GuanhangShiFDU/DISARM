# DISARM

DISARM is a training-free runtime defense against indirect prompt injection in
tool-augmented LLM agents. It uses a lightweight taint gate to select risky
observation-action checkpoints, performs one counterfactual replay after
ablating suspicious spans, and enforces an `ALLOW`, `RECOVER`, or `BLOCK`
decision from action-level trace divergence.

This repository is intentionally a small overlay for AgentDojo rather than a
fork of the full benchmark. The included adapter is pinned to a reviewed
AgentDojo revision and keeps the integration diff explicit.

## Contents

- `disarm/`: the DISARM executor, taint detection and neutralization logic.
- `agentdojo_adapter/install.py`: installer for a clean AgentDojo checkout.
- `agentdojo_adapter/agentdojo-v0.1.35.patch`: minimal pipeline, metric, and
  trace-persistence integration.
- `tests/`: offline regression tests for SAGE, counterfactual enforcement,
  structured observations, encoded content, and multi-action batches.

## Installation

Clone DISARM and AgentDojo as sibling directories:

```bash
git clone https://github.com/GuanhangShiFDU/DISARM.git
git clone https://github.com/ethz-spylab/agentdojo.git

cd agentdojo
git checkout 5cea5891fa8e6b13c4299a94691e1ec64d445fcd
python ../DISARM/agentdojo_adapter/install.py --dry-run .
python ../DISARM/agentdojo_adapter/install.py .
uv sync
```

The installer refuses unsupported or dirty AgentDojo checkouts by default. Use
`--allow-unsupported` or `--allow-dirty` only after reviewing the integration
patch.

Set the provider key in your shell or in AgentDojo's local `.env` file:

```bash
export OPENAI_API_KEY="..."
```

## Smoke test

Run one AgentDojo case:

```bash
uv run python -m agentdojo.scripts.benchmark \
  --model gpt-4o-2024-05-13 \
  --attack important_instructions \
  --defense disarm \
  --suite slack \
  --user-task user_task_0 \
  --injection-task injection_task_0 \
  --logdir runs/disarm-smoke
```

Run all four suites under Tool Knowledge:

```bash
uv run python -m agentdojo.scripts.benchmark \
  --model gpt-4o-2024-05-13 \
  --attack tool_knowledge \
  --defense disarm \
  --benchmark-version v1.2.2 \
  --suite banking --suite slack --suite travel --suite workspace \
  --max-workers 1 \
  --logdir runs/disarm-tool-knowledge
```

Each AgentDojo trace includes `input_tokens`, `output_tokens`, and a
`defense_metrics` object. DISARM audit events contain the SAGE evidence, taint
closure, factual and shadow action batches, TDV fields, verdict, reason, and
executed action batch.

## Configuration

The paper configuration is the default:

```bash
export DISARM_AUDIT_POLICY=selective
export DISARM_ENFORCEMENT_POLICY=differential
export DISARM_DEBUG=false
```

Two experiment-only ablations are available:

- `DISARM_AUDIT_POLICY=always` bypasses only the selective SAGE gate while
  retaining the same counterfactual audit and enforcement logic.
- `DISARM_ENFORCEMENT_POLICY=gate_only_block` blocks gated actions without
  executing the counterfactual audit.

## Offline tests

After installing the overlay into AgentDojo:

```bash
PYTHONPATH=src uv run pytest -q \
  ../DISARM/tests/test_disarm.py \
  ../DISARM/tests/test_disarm_neutralize.py
```

The tests do not make API calls.

## Reproducibility policy

API and transport failures may be retried from a fresh task state. Experimental
outcomes must not be retried or replaced based on utility or attack success.
Record the exact model identifier, AgentDojo commit, benchmark version, DISARM
configuration, and failed-case count with every reported table.

## Status

The paper is under review. This branch is a revision release candidate; final
paper-table scripts and aggregate results will be added only after the reported
configuration and implementation are frozen.

## License

See `LICENSE`.
