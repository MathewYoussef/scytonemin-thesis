# Auditor A3 — Implementation (Step-by-Step)

## 0) Scope enforcement
- Verify changed paths are within allowed patterns.

## 1) Validate catalog
- Confirm `ops/output/data/catalog.csv` exists and includes new assets.

## 2) Check schemas & provenance
- For each `data/<dataset_id>/`, ensure `README.md`, `schema.json`, and `provenance.yaml` exist.

## 3) Sample size checks
- Verify `data-sample/**` files are ≤ 10 MB each (list and sum sizes).

## 4) Processed tables & checksums
- Spot-check that processed tables have matching schema columns and checksum files.

## 5) Figures under scaffold
- Verify presence of images in `scaffold/<block>/figures/**`.

## 6) Large files policy
- Ensure raw heavy data is via DVC/LFS (presence of pointer files).

## 7) Write review
- Use `ops/auditors/templates/PR_REVIEW_A3.md` to report findings.
