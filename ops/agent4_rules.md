# Agent 4 — Rules: Media Placement & Reviewer Experience (Images & Video)

**Mission**: Decide and implement the placement of images and videos so a reviewer can visually audit the experiment and setup. Curate captions, alt text, thumbnails, and page locations aligned to Agent 2’s docs scaffold and Agent 3’s catalogs. Do **not** move files unrelated to media or restructure source code/data.

---

## Scope
- Media inventory comes from `Supplements/` (images & videos):
  - Timelapses: nostoc triple soak cleaning; nostoc measurements (spectroscopy setup); rotovap & analyte extraction
  - Photos: shaler table, tent, table used, white references (rows & gradient; UV on/off), lyophilizer, post‑exposure vialing, rehydration pipetting, diffusing plate before/after (50×50→5×5 cm), off‑nadir box section, wet nostoc colony at 4 °C
- Produce a **media map** that ties each asset → page location → caption → alt text → claim(s) supported → license.
- Place **low‑weight derivatives** (thumbnails, web videos) under `docs/media/**` and write pages linking to high‑res originals managed by Agent 3 (DVC/LFS).
- Ensure accessibility (alt text), compression, and coherent narrative order.

## Non‑interference contract
- Write only under: `docs/media/**`, `docs/**/media.md`, and `ops/output/media/**` (indexes).  
- Do not write into `scaffold/**` (Agent 2’s build outputs) nor `data/**` (Agent 3).  
- Branch: `agent4/media-storyboard`.

## Inputs
- `ops/output/inventory/inventory.csv` (Agent 1)
- `ops/output/data/catalog.csv` (Agent 3)
- Scaffold docs structure (`docs/**`) from Agent 2

## Required Deliverables
1. **Media map**: `ops/output/media/media_map.csv` with columns:  
   `asset_relpath,block,page,section,caption,alt_text,license,thumb_relpath,poster_relpath,claim_ids,priority`
2. **Galleries**: `docs/<block>/media.md` with image grids and video embeds.
3. **Thumbnails/posters** in `docs/media/thumbs/` and `docs/media/posters/` (derivatives, small).
4. **Placement guide**: `docs/contrib/media_guide.md` (how to add new assets).

## Acceptance Criteria
- Every listed media asset has alt text, caption, and a page destination.
- Pages render with reasonable load (thumbnails rather than raw 4K/RAW files).
- No large originals are embedded directly in docs; link to DVC/LFS where needed.

## Checklist
- [ ] Catalog pulled
- [ ] Media map drafted
- [ ] Thumbs/posters generated
- [ ] Block pages created
- [ ] PR opened: **media experience**
