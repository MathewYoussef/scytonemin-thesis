# Agent 2 — Rules: Scaffolding & Presentation (Docs, Structure, Aesthetics)

**Mission**: Create all scaffolds for the thesis audit experience: directory structure, docs site, figure notebook placeholders, plot/text integration points, and a clear visual map for where data, images, and videos will live. Do **not** reorganize existing files or harvest data/media.

---

## Scope
- Build a **non-invasive** scaffold under `scaffold/` with readme files describing what will go where.
- Create a MkDocs site skeleton (`docs/`, `mkdocs.yml`) and navigation aligned to blocks:
  - Hub (landing, How to Audit in 10 Minutes, Code & Data Availability)
  - Reflectance
  - Initial Calibration (Chromatogram→DAD→Concentration→Dry Weight)
  - Act of God — Mamba‑SSM
  - Supplements (calibrations, UV dose, instruments, images, videos)
- Create placeholders for: `figures/`, `tables/`, `notebooks/`, `assets/images/`, `assets/videos/`.
- Define **plot integration points** (where notebooks will write PNG/SVG/PDF) and **text integration points** (where thesis excerpts or captions live).
- Prepare `CLAIMS.md` layout & link anchors (do not populate claims beyond placeholders).
- Do not create or modify files under `ops/output/**` (Agent 1), `data/**` (Agent 3), `docs/media/**` (Agent 4).

## Non‑interference contract
- Write only under `scaffold/**` and `docs/**` (content pages & placeholders), `mkdocs.yml`, and `Makefile` stubs.
- Never move or delete any existing file. Your changes must be additive.
- Branch: `agent2/scaffold-hub`.

## Inputs
- `ops/output/inventory/inventory.csv` (read‑only).
- `ops/output/proposals/mapping.csv` (read‑only).
- `thesis.md` (read‑only) for text snippets & headings.

## Required Deliverables
1. `mkdocs.yml` (Material theme skeleton) and `docs/index.md` (audit how‑to).
2. `scaffold/` tree with `README.md` files in each subdir describing intended content/location.
3. `docs/claims/index.md` and per‑claim page template (`docs/claims/C-XXX.md`).
4. `docs/reflectance/`, `docs/initial_calibration/`, `docs/mamba_ssm/`, `docs/supplements/` with page templates.
5. `Makefile` targets: `quickstart`, `figures`, `audit` (stubs only).

## Acceptance Criteria
- MkDocs builds locally (`mkdocs build`) with no broken nav.
- All scaffold folders include a one‑page README explaining purpose and write API for Agents 3 & 4.
- No collisions with existing paths (everything additive).

## Checklist
- [ ] `mkdocs.yml` created
- [ ] `docs/` section pages & templates created
- [ ] `scaffold/` directories + readmes created
- [ ] `Makefile` stubs added
- [ ] PR opened: **scaffold‑only**
