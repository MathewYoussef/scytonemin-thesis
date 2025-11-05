# Agent 2 — Implementation Plan (Step‑by‑Step)

> Branch: `agent2/scaffold-hub`  
> Write scope: `docs/**`, `scaffold/**`, `mkdocs.yml`, `Makefile` only.

## 0) Bootstrap
```bash
git checkout -b agent2/scaffold-hub
```

## 1) MkDocs skeleton
```bash
cat > mkdocs.yml <<'YAML'
site_name: Scytonemin Thesis — Audit Hub
nav:
  - Home: index.md
  - Claims:
    - claims/index.md
  - Reflectance:
    - reflectance/index.md
  - Initial Calibration:
    - initial_calibration/index.md
  - Mamba‑SSM:
    - mamba_ssm/index.md
  - Supplements:
    - supplements/index.md
theme:
  name: material
markdown_extensions:
  - toc:
      permalink: true
YAML
mkdir -p docs/{claims,reflectance,initial_calibration,mamba_ssm,supplements}
```

Create minimal pages (each with 2–3 paragraphs explaining what will appear, links to scaffold locations, and placeholders for images/plots).

## 2) Scaffold directories
```bash
mkdir -p scaffold/{hub,reflectance,initial_calibration,mamba_ssm,supplements}/{figures,tables,notebooks,assets/images,assets/videos}
for d in hub reflectance initial_calibration mamba_ssm supplements; do
  cat > scaffold/$d/README.md <<'MD'
# Scaffold — Purpose & Write API
- **Purpose:** Placeholder destination for generated figures/tables/notebooks and curated media for this block.
- **Writers:** Agent 3 (data/analysis outputs), Agent 4 (media placements).
- **Do not** commit large raw data here; Agent 3 will manage via DVC/LFS and link processed artifacts here.
- **Expected outputs:** see docs pages in `docs/<block>/` for required assets and naming.
MD
done
```

## 3) Makefile stubs
```bash
cat > Makefile <<'MAKE'
.PHONY: quickstart figures audit docs
quickstart:
	@echo "Run minimal notebooks on sample data (Agent 3 to provide)"
figures:
	@echo "Build all figures into scaffold/*/figures"
audit:
	@echo "Run claim checks (to be implemented in CI)"
docs:
	mkdocs build -q
MAKE
```

## 4) Claims templates
- `docs/claims/index.md`: overview of how claims map to notebooks/data/tests.
- Template `docs/claims/C-XXX.md` explains required links: notebook cell, data paths (with checksums), code path, and test name.
