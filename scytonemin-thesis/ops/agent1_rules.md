# Agent 1 — Rules: Inventory & Reorganization Lead

**Mission**: Clone the repository and produce a **complete, lossless inventory and reorganization proposal** for `IMAS-portfolio` that separates unrelated projects, prepares subprojects, and creates a clean foundation for Agents 2–4. Do **not** create decorative scaffolds, fetch external data, or place media.

---

## Scope
- Clone and work against: `https://github.com/MathewYoussef/IMAS-portfolio`.
- Produce a **canonical inventory** (CSV + JSON) of every trackable path in the repo.
- Classify each path into a taxonomy: `code`, `notebooks`, `docs`, `data/raw`, `data/processed`, `media`, `results`, `config`, `env`, `vendor-exports`, `bin`, `misc`.
- Detect **unrelated projects** (by topic/owner/toolchain) and propose extraction into `subprojects/` or external repos.
- Propose a **target tree** that aligns with blocks: `Reflectance`, `Initial_Calibration`, `Act_of_God_Mamba_Results`, `Supplements`, and a lightweight **hub**.
- Identify **oversized files** (>90MB), recommend `git-lfs` or `DVC`.
- Identify duplicates, orphaned outputs, broken links, and dead notebooks.
- **Do not** alter or create files under `scaffold/` (owned by Agent 2), `data/` (owned by Agent 3), or `docs/media/` (owned by Agent 4).

## Non‑interference contract
- Write *only* under:
  - `ops/output/inventory/` (reports)
  - `ops/output/proposals/` (mapping tables & restructure proposal)
  - `ops/logs/` (run logs)
- Do not move, rename, or delete any tracked file in the first PR. Your first PR is **reports‑only**.
- Use a dedicated branch: `agent1/inventory-restructure`.

## Inputs
- The current repo contents (do not assume any pre-knowledge of what exists).
- `thesis.md` (if present), purely as context for topic grouping (no edits).

## Required Deliverables
1. **Tree report**: `ops/output/inventory/tree.txt` (depth unlimited; excludes `.git`).
2. **Inventory**: `ops/output/inventory/inventory.csv` with columns:  
   `path,type,size_bytes,sha1,last_commit,language,category,block_guess,action_suggestion,new_location,notes`
3. **Duplicates**: `ops/output/inventory/duplicates.csv` (sha1, paths).
4. **Large files**: `ops/output/inventory/large_files.csv` (>90MB).
5. **Mapping table**: `ops/output/proposals/mapping.csv` (old_path → proposed_new_path → rationale → risk).
6. **Restructure proposal**: `ops/output/proposals/restructure_proposal.md` (target tree + rationale + risks).
7. **Issue list**: `ops/output/proposals/issues.md` (per-path issues to open).

## Acceptance Criteria
- Inventory covers **100%** of tracked files/folders (excluding `.git/`).
- Every path has a category and proposed action (`keep`, `move`, `extract`, `archive`, `delete?`).
- No writes outside `ops/` directories. CI passes (lint on CSV/MD).

## Checklist
- [ ] Repo cloned & branch created
- [ ] `tree.txt` generated
- [ ] `inventory.csv` complete
- [ ] `duplicates.csv` & `large_files.csv` complete
- [ ] `mapping.csv` drafted
- [ ] `restructure_proposal.md` drafted
- [ ] PR opened: **reports‑only**
