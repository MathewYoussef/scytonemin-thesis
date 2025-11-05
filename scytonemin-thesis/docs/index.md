# Scytonemin Thesis Audit Hub

Welcome to the shared launchpad for auditors, maintainers, and co-authors working across the Scytonemin thesis artifacts. This site describes how the blocks fit together inside the consolidated `scytonemin-thesis/` repository: which directories own each deliverable, which pipelines regenerate the outputs, and where evidence for every thesis claim will land once all agents finish their passes.

## How to Audit in 10 Minutes

1. **Run `notebooks/00_env_and_schema.ipynb`** — freeze the Python environment, record dataset schemas, and capture a hash of `src/`. Keep the generated `reports/environment.txt` with any reviewer bundle.
2. **Review the artefact map** — Each block page lists the raw + reference directories (`data/raw/**`, `data/reference/**`) and the scripts that regenerate them. Check the README + checksum manifests to verify provenance.
3. **Run the pipelines** — Use `make reproduce` to rebuild Stage A/B (calibration) and the reflectance canonical bundle. The Mamba evaluation stage is optional and will only run if Torch/mamba-ssm are installed.
4. **Verify the claims ledger** — The `Claims` section links every thesis statement to canonical data products, scaffold figures, and (soon) automated tests.

## Responsibilities & Handoffs

- **Agent 2 (Scaffolding)** — Owns the documentation, scaffold directories, and pipeline wiring (`Makefile`, `src/pipelines`).
- **Agent 3 (Data & Analysis)** — Extends the pipelines (tests, notebooks) and ensures regenerated outputs stay in sync with checksum manifests.
- **Agent 4 (Media)** — Curates photos/videos through `docs/media/**` and links them to block narratives.
- **Thesis manuscript** — `thesis.docx.md` (repo root) is the canonical source for prose excerpts. Use the headings in that file when filling each page’s “Thesis integration points” checklist.

## Integration Checklist

- `Makefile` exposes working targets: `make setup`, `make reproduce`, `make quickstart`, `make docs`, and (soon) `make audit` for regression tests.
- `mkdocs build` should run cleanly at every commit; keep navigation aligned with the `scaffold/` layout so reviewers can trace evidence quickly.
- `ops/logs/` records each migration or rerun; add a new dated entry whenever raw assets or evaluation scripts change.
