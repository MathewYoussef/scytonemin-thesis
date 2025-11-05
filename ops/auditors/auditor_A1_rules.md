# Auditor A1 — Rules (Audits Agent 1: Inventory & Reorganization)

**Mission:** Verify Agent 1's reports are complete, non-destructive, and follow scope. Confirm 100% coverage of tracked files and that proposals are actionable. No file moves/deletions occur in the PR.


**Path ownership (must enforce):**
- Agent 1 may write **only** under: `ops/output/**`, `ops/logs/**`.
- Agent 2 may write **only** under: `docs/**`, `scaffold/**`, `mkdocs.yml`, `Makefile`.
- Agent 3 may write **only** under: `data/**`, `models/**`, `ops/output/data/**`, and `scaffold/**/figures`.
- Agent 4 may write **only** under: `docs/media/**`, `docs/**/media.md`, `ops/output/media/**`.


## Inputs
- PR labeled `A1`
- Artifacts expected from Agent 1:
  - `ops/output/inventory/tree.txt`
  - `ops/output/inventory/inventory.csv`
  - `ops/output/inventory/duplicates.csv`
  - `ops/output/inventory/large_files.csv`
  - `ops/output/proposals/mapping.csv`
  - `ops/output/proposals/restructure_proposal.md`
  - `ops/output/proposals/issues.md`

## What to check
1. **Scope:** Changed files restricted to `ops/output/**` and `ops/logs/**`.
2. **Completeness:** `inventory.csv` has required columns and row count ≈ file count from `tree.txt` (excluding `.git`).  
3. **Classification quality:** No empty `category` or `action_suggestion` cells after header.  
4. **Large file policy:** `large_files.csv` includes all files > 90MB.  
5. **Duplicates:** `duplicates.csv` shows any sha1 duplicates; if none found, file still present with only header.
6. **Mapping:** `mapping.csv` has `old_path,new_path,action,rationale,risk` headers and at least N entries for out-of-place items.
7. **Proposal:** `restructure_proposal.md` contains target tree, risks, and extraction plan for unrelated projects.
8. **No Destructive Ops:** No deletes/renames in this PR.

## Acceptance criteria
- All checks pass. No out-of-scope changes. Missing items are flagged with line comments using the template below.

## Deliverables (by auditor)
- `ops/auditors/reports/A1_review.md` (filled template)
- CI status green for `audit-agent1` on the PR
