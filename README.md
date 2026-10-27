# DISARM

Reference implementation of **DISARM: Taint-Gated Counterfactual Auditing for Efficient Defense against Indirect Prompt Injection**.

DISARM is a training-free defense for tool-augmented LLM agents. It selectively audits suspicious tool observations, constructs a taint-ablated counterfactual view, and applies an **ALLOW / RECOVER / BLOCK** decision at the tool boundary.

Paper: [Journal of Computer Security](https://doi.org/10.1177/0926227X261505043)

## Repository layout

```text
disarm/                         Core DISARM implementation
agentdojo_adapter/
  install.py                    Installer for a clean AgentDojo checkout
  agentdojo-v0.1.35.patch       Minimal AgentDojo integration patch
  files/metrics.py              Runtime accounting used by the adapter
tests/                          Offline unit tests
```

The repository is distributed as a small overlay rather than a fork of AgentDojo. The installer applies the integration patch and copies the DISARM modules into a pinned AgentDojo checkout.

## Requirements

- Git
- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- An OpenAI API key for the GPT-4o reproduction commands below

No separate virtual environment needs to be created manually; `uv sync` creates and manages it.

## Installation

Clone DISARM and AgentDojo into the same parent directory:

```bash
git clone https://github.com/GuanhangShiFDU/DISARM.git
git clone https://github.com/ethz-spylab/agentdojo.git

cd agentdojo
git checkout 5cea5891fa8e6b13c4299a94691e1ec64d445fcd
```

The commit above is AgentDojo v0.1.35, the version supported by the adapter. From the AgentDojo directory, validate and install the overlay:

```bash
python3 ../DISARM/agentdojo_adapter/install.py . --dry-run
python3 ../DISARM/agentdojo_adapter/install.py .
uv sync
```

The installer expects a clean AgentDojo checkout. It will stop rather than overwrite unrelated local changes.

## API configuration

Create `agentdojo/.env`:

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

AgentDojo loads this file when the benchmark command starts. Do not commit the file.

## Verify the installation

Run the offline tests from the AgentDojo directory:

```bash
uv run pytest ../DISARM/tests
```

Then run one attacked task as a smoke test:

```bash
uv run python -m agentdojo.scripts.benchmark \
  --model gpt-4o-2024-05-13 \
  --benchmark-version v1.2.2 \
  --suite slack \
  --user-task user_task_0 \
  --injection-task injection_task_0 \
  --attack important_instructions \
  --defense disarm \
  --logdir runs/disarm-smoke \
  --force-rerun
```

The command should report `Using defense: 'disarm'` and write the task trace under `runs/disarm-smoke/`.

## Reproducing the AgentDojo evaluation

The following commands use GPT-4o (`gpt-4o-2024-05-13`), AgentDojo benchmark version `v1.2.2`, and all four suites used in the paper: Banking, Slack, Travel, and Workspace.

These runs call a paid model API. Start with the smoke test and estimate cost before launching the complete matrix.

### Benign utility

Run DISARM without an attack:

```bash
uv run python -m agentdojo.scripts.benchmark \
  --model gpt-4o-2024-05-13 \
  --benchmark-version v1.2.2 \
  --defense disarm \
  --suite banking \
  --suite slack \
  --suite travel \
  --suite workspace \
  --max-workers 1 \
  --logdir runs/disarm-no-attack
```

### Four-attack evaluation

Run the four AgentDojo attacks reported in the paper:

```bash
for attack in ignore_previous system_message important_instructions tool_knowledge; do
  uv run python -m agentdojo.scripts.benchmark \
    --model gpt-4o-2024-05-13 \
    --benchmark-version v1.2.2 \
    --attack "$attack" \
    --defense disarm \
    --suite banking \
    --suite slack \
    --suite travel \
    --suite workspace \
    --max-workers 1 \
    --logdir "runs/disarm-${attack}"
done
```

AgentDojo resumes from existing task traces when `--force-rerun` is omitted. Add `--force-rerun` only when you intentionally want to replace completed traces.

## Output and metrics

Each `--logdir` contains AgentDojo JSON traces organized by model, suite, user task, attack, and injection task. In addition to AgentDojo's utility and security outcomes, the adapter records:

- input and output token counts;
- total and auxiliary LLM calls;
- SAGE gate checks and audit events;
- ALLOW, RECOVER, and BLOCK verdicts;
- tool-path and critical-argument comparison evidence.

The corresponding fields are stored in `input_tokens`, `output_tokens`, and `defense_metrics` in each trace.

## Optional ablations and debugging

The default paper configuration uses selective auditing and differential enforcement. These environment variables expose the ablations used by the implementation:

```bash
# Audit every checkpoint instead of using the SAGE gate.
export DISARM_AUDIT_POLICY=always

# Block directly on a gate trigger without counterfactual enforcement.
export DISARM_ENFORCEMENT_POLICY=gate_only_block

# Print detailed DISARM diagnostics.
export DISARM_DEBUG=1
```

Unset the variables to restore the default configuration:

```bash
unset DISARM_AUDIT_POLICY DISARM_ENFORCEMENT_POLICY DISARM_DEBUG
```

## Troubleshooting

- **Unsupported AgentDojo commit:** check out `5cea5891fa8e6b13c4299a94691e1ec64d445fcd`. Use `--allow-unsupported` only after manually reviewing that the patch still applies correctly.
- **Dirty AgentDojo checkout:** install into a fresh checkout, or review the changes before using `--allow-dirty`.
- **SOCKS proxy error:** if `ALL_PROXY` or `HTTPS_PROXY` uses a SOCKS URL, install SOCKS support with `uv pip install "httpx[socks]"`.
- **Existing traces are skipped:** this is expected resume behavior. Use a new `--logdir`, or add `--force-rerun` when replacement is intended.

## License

This project is released under the [MIT License](LICENSE).
