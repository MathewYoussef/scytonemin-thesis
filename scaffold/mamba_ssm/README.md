# Mamba-SSM Scaffold — Purpose & Write API

**Purpose:** Stage evaluation artifacts, diagnostics, and notebooks for the Act-of-God warm-start experiment.

## Writers
- **Agent 3:** Exports validation metrics, manifest tables, and checkpoint summaries from `Act_of_God_Mamba_Results/`.
- **Agent 4:** Adds explanatory media (e.g., workflow animations) before promoting derivatives to the docs site.

## Expected Outputs
- `figures/` — Loss curves, readiness gate dashboards, downstream proxy comparisons.
- `tables/` — CSV/Parquet summaries (per-gate evaluation, manifest counts, checkpoint metadata).
- `notebooks/` — Evaluation notebooks or scripts that reproduce AoG results.
- `assets/images/`, `assets/videos/` — Temporary media staging for this block.

## Conventions
- Reference checkpoint artifacts via DVC/LFS metadata rather than embedding binaries.
- Include commit hashes and config versions in notebooks/tables for provenance.
- Follow the naming pattern `aog_<descriptor>.<ext>` to keep assets grouped.

