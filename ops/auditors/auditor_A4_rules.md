# Auditor A4 — Rules (Audits Agent 4: Media Placement & Reviewer Experience)

**Mission:** Verify Agent 4's media plan is accessible, fast to load, properly captioned, and scoped to allowed paths. Ensure media pages are created per block and link to high-res originals managed by Agent 3.


**Path ownership (must enforce):**
- Agent 1 may write **only** under: `ops/output/**`, `ops/logs/**`.
- Agent 2 may write **only** under: `docs/**`, `scaffold/**`, `mkdocs.yml`, `Makefile`.
- Agent 3 may write **only** under: `data/**`, `models/**`, `ops/output/data/**`, and `scaffold/**/figures`.
- Agent 4 may write **only** under: `docs/media/**`, `docs/**/media.md`, `ops/output/media/**`.


## Inputs
- PR labeled `A4`
- Artifacts expected from Agent 4:
  - `ops/output/media/media_map.csv`
  - `docs/<block>/media.md` per block
  - Thumbnails/posters in `docs/media/{thumbs,posters}/`
  - `docs/contrib/media_guide.md`

## What to check
1. **Scope:** Changed files restricted to `docs/media/**`, `docs/**/media.md`, `ops/output/media/**`.
2. **Media map quality:** Columns present (`asset_relpath,block,page,section,caption,alt_text,license,thumb_relpath,poster_relpath,claim_ids,priority`); no empty `alt_text` or `caption`.
3. **Derivatives present:** Thumbnails/posters exist and are referenced.
4. **Page performance:** Pages embed thumbnails, not huge originals; links to originals okay.
5. **Licensing:** Media map includes license field for each asset.

## Acceptance criteria
- All media assets have alt text and captions; pages render quickly; no scope violations.

## Deliverables (by auditor)
- `ops/auditors/reports/A4_review.md` (filled template)
- CI status green for `audit-agent4` on the PR
