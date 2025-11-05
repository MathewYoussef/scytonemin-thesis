# Repository‑wide Gameplan — Scytonemin Thesis Audit Build‑out

**Objective:** Transform `IMAS-portfolio` into a reviewer‑friendly, reproducible thesis audit hub without losing history or stepping on parallel work.

---

## Phases & Concurrency Model
All agents work **in parallel** on separate branches and path namespaces.

- **Agent 1 — Inventory & Reorganization**  
  Branch: `agent1/inventory-restructure`  
  Paths: `ops/output/**`, `ops/logs/**` only

- **Agent 2 — Scaffolding & Presentation**  
  Branch: `agent2/scaffold-hub`  
  Paths: `docs/**`, `scaffold/**`, `mkdocs.yml`, `Makefile`

- **Agent 3 — Data & Analysis Capture**  
  Branch: `agent3/data-harvest-setup`  
  Paths: `data/**`, `models/**`, `ops/output/data/**`, `scaffold/**/figures`

- **Agent 4 — Media Placement**  
  Branch: `agent4/media-storyboard`  
  Paths: `docs/media/**`, `docs/**/media.md`, `ops/output/media/**`

**Ground rules:**
- No agent edits another’s paths. All changes are additive until integration PRs.
- Integration happens via **PR chaining**:
  1) Agent 1 PR (reports) → 2) Agent 2 PR (scaffold) → 3) Agent 3 PR (data/figures) → 4) Agent 4 PR (media).
- If a dependency is missing, proceed with stubs and note the gap in your checklist.

---

## Target Information Architecture (top‑level)
```
hub/                      # eventual hub (may start as docs site)
Reflectance/              # code, notebooks, processed data (Agent 3 curated)
Initial_Calibration/      # chromatogram/DAD→concentration→dry weight
Act_of_God_Mamba_Results/ # Mamba‑SSM models, configs, results
Supplements/              # calibrations, UV dose, instruments, images/videos (raw)
scaffold/                 # Agent 2 destinations for figures/tables/notebooks/media
data*/ models*/           # Agent 3 managed (DVC/LFS)
docs/                     # Docs site (Agent 2), media pages (Agent 4)
ops/                      # Reports, catalogs, proposals, logs
subprojects/              # Unrelated projects extracted here (Agent 1 proposal)
archive/                  # Exiled legacy artifacts (no deletion without review)
```

---

## Branch, PR, and Ownership
- **CODEOWNERS** (to add later):
  - `docs/**` → Agent 2 + Owner
  - `scaffold/**` → Agent 2
  - `data/**`, `models/**` → Agent 3
  - `docs/media/**` → Agent 4
  - `ops/**` → Agent 1
- **PR labels:** `A1`, `A2`, `A3`, `A4`, `integration`
- **Commit message pattern:** `A<N>: <short action> (path)`

---

## Checklists (global)
- [ ] Clone repo locally and set upstream
- [ ] Create branches per agent
- [ ] Agent 1 inventory & proposal merged
- [ ] Agent 2 scaffold merged
- [ ] Agent 3 data & figures merged
- [ ] Agent 4 media experience merged
- [ ] Tag `v0.1.0` smoke build
- [ ] Archive to Zenodo and tag `v1.0.0`

---

## Risks & Mitigations
- **Large files in history** → Mitigate with LFS/DVC; do not rewrite history now.
- **Orphaned outputs** → Catalog via Agent 1; either re‑generate or archive.
- **Cross‑agent path collisions** → Prevent with path ownership + branch isolation.
- **Missing licenses** → Default to `UNKNOWN`, open issues to set correct license per dataset/media.

---

## Quickstart (operators)
```bash
git clone https://github.com/MathewYoussef/IMAS-portfolio
cd IMAS-portfolio
# Read ops/agents/* for your role, then create your branch and begin.
```
