# PR: Agent 2 — Scaffolding & Presentation (docs + scaffold)

## Summary
Additive scaffolding for the thesis audit hub:
- MkDocs Material site with landing, Code & Data Availability, block pages, and Claims.
- Complete scaffold/ directory tree with write‑API READMEs for each block (hub, reflectance, initial_calibration, mamba_ssm, supplements).
- Claims ledger overview + template, plus instantiated claims C‑001, C‑002, C‑003, C‑004, C‑005, C‑006, C‑007, C‑008, C‑009, C‑010, C‑011, C‑012 anchored to manuscript or AoG repo.
- Makefile stubs for quickstart/figures/audit/docs.

## Paths Touched (Agent‑2‑owned only)
- mkdocs.yml
- Makefile
- docs/index.md, docs/availability.md
- docs/claims/index.md, docs/claims/C-XXX.md
- docs/claims/C-001.md … C-012.md
- docs/reflectance/index.md, docs/initial_calibration/index.md, docs/mamba_ssm/index.md, docs/supplements/index.md
- scaffold/**/README.md and subfolders (figures, tables, notebooks, assets/images, assets/videos)

## Acceptance Criteria
- MkDocs builds locally with no broken nav.
- All scaffold folders include a one‑page README with purpose and write API for Agents 3 & 4.
- Changes are additive and confined to Agent‑2 scope.

## Build
- Install MkDocs Material (one‑off): `python3 -m pip install --user --break-system-packages mkdocs-material`
- Build docs: `make docs` (or `python3 -m mkdocs build`)

## Non‑Interference
- No edits to `ops/output/**`, `data/**`, or `docs/media/**`.
- No reorganization of legacy folders; scaffolding is additive only.

## Next Steps (post‑merge)
- Agent 3: begin writing processed tables/figures to `scaffold/<block>/**` and publish sample datasets; add DVC/LFS pointers.
- Agent 4: curate media derivatives under `docs/media/**` and link to block pages.
- Authoring: expand manuscript excerpts in block pages; flesh out claim pages with notebook cells, data checksums, and test names.

## Checklist
- [x] mkdocs.yml created and nav wired
- [x] docs/ section pages & templates created
- [x] scaffold/ directories + readmes created
- [x] Makefile stubs added
- [x] Claims pages added and linked

