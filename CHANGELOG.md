# Changelog

## Unreleased revision candidate

Compared with the initial public snapshot, this candidate adds:

- complete factual and counterfactual action-batch comparison;
- explicit selective-audit and always-audit policies;
- differential and gate-only enforcement modes for ablation;
- structured audit records with SAGE evidence, taint spans, TDV, verdicts, and
  executed actions;
- conservative recovery-application failure handling;
- JSON, YAML, embedded structured-object, encoded-content, and correlated
  cross-observation taint handling;
- token and auxiliary-call accounting hooks for AgentDojo traces;
- offline regression tests and a version-checked AgentDojo installer.

The initial submission implementation should be preserved with a release tag
before this candidate is merged into the public default branch.
