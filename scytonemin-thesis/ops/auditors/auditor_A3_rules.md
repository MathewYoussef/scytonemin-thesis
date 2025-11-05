# Auditor A3 — Rules (Audits Agent 3: Data & Analysis Capture)

**Mission:** Verify Agent 3's data capture is reproducible, versioned, lightweight for quickstart, and scoped to allowed paths. Ensure figures render into the scaffold; ensure DVC/LFS is configured.


**Path ownership (must enforce):**
- Agent 1 may write **only** under: `ops/output/**`, `ops/logs/**`.
- Agent 2 may write **only** under: `docs/**`, `scaffold/**`, `mkdocs.yml`, `Makefile`.
- Agent 3 may write **only** under: `data/**`, `models/**`, `ops/output/data/**`, and `scaffold/**/figures`.
- Agent 4 may write **only** under: `docs/media/**`, `docs/**/media.md`, `ops/output/media/**`.


## Inputs
- PR labeled `A3`
- Artifacts expected from Agent 3:
  - DVC init (`.dvc/`, `dvc.yaml`) or LFS setup
  - `ops/output/data/catalog.csv`
  - Data dictionaries: `data/<dataset_id>/README.md` and `schema.json`
  - Provenance: `data/<dataset_id>/provenance.yaml`
  - Samples: `data-sample/<dataset_id>/*` (≤10 MB each)
  - Processed tables: `data-processed/<block>/**` (+ checksums)
  - Figures written to `scaffold/<block>/figures/**`
  - `ops/output/data/checks.md`

## What to check
1. **Scope:** Changed files restricted to `data/**`, `models/**`, `ops/output/data/**`, and `scaffold/**/figures`.
2. **Catalog completeness:** Catalog entries exist for all new datasets/models in the PR.
3. **Schema & provenance:** Each dataset has `schema.json` and `provenance.yaml`.
4. **Sample size:** Each sample dataset ≤ 10 MB; quickstart feasible.
5. **Processed tables:** Include checksums; columns match schema.
6. **Figures present:** At least the flagship figures exist under scaffold paths.
7. **No large raw blobs:** Large files should be tracked via DVC/LFS, not committed to git.

## Acceptance criteria
- All items present; quickstart feasible; no scope violations.

## Deliverables (by auditor)
- `ops/auditors/reports/A3_review.md` (filled template)
- CI status green for `audit-agent3` on the PR
