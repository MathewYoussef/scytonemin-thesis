# Auditor A2 — Rules (Audits Agent 2: Scaffolding & Presentation)

**Mission:** Verify Agent 2's PR is additive, builds the docs site, and provides clear landing zones for Agents 3 and 4. No data/media ingestion here.


**Path ownership (must enforce):**
- Agent 1 may write **only** under: `ops/output/**`, `ops/logs/**`.
- Agent 2 may write **only** under: `docs/**`, `scaffold/**`, `mkdocs.yml`, `Makefile`.
- Agent 3 may write **only** under: `data/**`, `models/**`, `ops/output/data/**`, and `scaffold/**/figures`.
- Agent 4 may write **only** under: `docs/media/**`, `docs/**/media.md`, `ops/output/media/**`.


## Inputs
- PR labeled `A2`
- Artifacts expected from Agent 2:
  - `mkdocs.yml`
  - `docs/index.md`, `docs/claims/index.md`
  - `docs/reflectance/index.md`, `docs/initial_calibration/index.md`, `docs/mamba_ssm/index.md`, `docs/supplements/index.md`
  - `scaffold/**/README.md` in each block
  - `Makefile` stubs

## What to check
1. **Scope:** Changed files restricted to `docs/**`, `scaffold/**`, `mkdocs.yml`, `Makefile`.
2. **MkDocs build:** `mkdocs build` completes with no errors.
3. **Navigation:** All listed pages exist and appear in nav; no broken links to non-existent scaffold paths.
4. **Scaffold READMEs:** Each block README explains purpose, expected writers, and naming conventions.
5. **Makefile stubs:** `quickstart`, `figures`, `audit`, `docs` targets exist.
6. **No data/media dumps:** Large binaries do not appear in this PR.

## Acceptance criteria
- Docs build succeeds; all scaffolds present; no out-of-scope changes.

## Deliverables (by auditor)
- `ops/auditors/reports/A2_review.md` (filled template)
- CI status green for `audit-agent2` on the PR
