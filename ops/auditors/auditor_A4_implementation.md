# Auditor A4 — Implementation (Step-by-Step)

## 0) Scope enforcement
- Verify changed paths are within allowed patterns.

## 1) Validate media_map.csv
- Ensure required columns exist.
- Fail if any row has empty `alt_text` or `caption`.

## 2) Thumbnails/posters
- Confirm derivative files exist under `docs/media/thumbs/` and `docs/media/posters/` and are referenced from pages.

## 3) Page checks
- Spot-check that block `media.md` pages use thumbnails and link to high-res originals (no raw 4K embeds).

## 4) Licensing
- Ensure `license` column populated; flag `UNKNOWN`.

## 5) Write review
- Use `ops/auditors/templates/PR_REVIEW_A4.md` to report findings.
