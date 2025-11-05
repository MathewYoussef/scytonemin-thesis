# Agent 4 — Implementation Plan (Step‑by‑Step)

> Branch: `agent4/media-storyboard`  
> Write scope: `docs/media/**`, `docs/**/media.md`, `ops/output/media/**` only.

## 0) Bootstrap
```bash
git checkout -b agent4/media-storyboard
mkdir -p ops/output/media docs/media/{thumbs,posters}
```

## 1) Build media inventory
- Start from `ops/output/data/catalog.csv` and filter assets under `Supplements/` with media extensions.
- If catalog missing, scan `Supplements/` for `*.png,*.jpg,*.jpeg,*.tif,*.tiff,*.mp4,*.mov,*.webm` and draft entries.

## 2) Create media map
- For each asset, assign a **block** and **docs page** section (from Agent 2 nav).
- Author **caption** and **alt text** tying the asset to claims or procedures.
- Decide **priority** (hero vs supporting) and whether a **thumb/poster** is needed.

## 3) Generate derivatives
- Produce thumbnails (~480px wide) and posters for videos (first salient frame).
- Save to `docs/media/thumbs/` and `docs/media/posters/` and reference in the media map.

## 4) Author block pages
- `docs/reflectance/media.md`, `docs/initial_calibration/media.md`, `docs/mamba_ssm/media.md`, `docs/supplements/media.md`
- Insert grids/galleries embedding thumbnails that link to high‑res (Agent 3‑managed) originals.

## 5) PR
- Ensure pages load fast and validate image paths.
- Open PR with media map, derivatives, and pages.
