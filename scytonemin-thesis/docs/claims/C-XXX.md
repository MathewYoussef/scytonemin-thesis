# Claim Template — C-XXX

> _Paste the thesis excerpt verbatim. Include citation markers._

## Evidence Checklist

- **Notebook cell:** `path/to/notebook.ipynb#cell-id`
- **Rendered figure/table:** `scaffold/hub/figures/<asset>.png`
- **Processed data:** `data-processed/<block>/<dataset>.csv @ <sha256>`
- **Source code:** `src/path/to/module.py @ <commit>`
- **Automated test:** `tests/test_<claim>.py::test_<scenario>`

## Reviewer Actions

1. Execute the referenced notebook cell and drop the artifact into the associated scaffold folder.
2. Confirm hashes and row counts of supporting datasets match the inventory in `ops/output/data/catalog.csv`.
3. Run `make audit` (once implemented) to validate the statistical guardrails for this claim.

## Notes

- Use `???` details while the target asset is missing. Replace placeholders with concrete paths before marking the claim complete.
- Media attachments (photos, videos) belong in `docs/media/` and should be linked from this page after Agent 4 publishes derivatives.

