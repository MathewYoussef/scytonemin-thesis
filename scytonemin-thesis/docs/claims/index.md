# Claims Ledger Overview

Every thesis statement referenced in the written document must resolve to reproducible evidence. The claims table captures the minimum metadata needed to audit a statement end-to-end.

## Structure

- **Claim pages** (`docs/claims/C-XXX.md`) declare the prose quote, context, and reviewer checks.
- **Generated ledger** — Agent 3 will build `scaffold/hub/tables/claims.csv` from the live Markdown files so we can diff evidence coverage.
- **Automation** — `make audit` will aggregate notebook assertions and data checks for each claim once implemented.

## Required Fields per Claim

| Field | Description | Who owns it |
| --- | --- | --- |
| Notebook Cell | Canonical execution cell that produces the figure/table supporting the claim. | Agent 3 |
| Data Inputs | Repository path + checksum for every referenced dataset. | Agent 3 |
| Code Path | Source file(s) or scripts generating the result. | Agent 3 |
| Tests | Automated validation covering the claim behaviour. | Agents 2 & 3 coordinate |
| Media | Optional photo/video evidence, referenced via `docs/media/**`. | Agent 4 |

## Open Tasks

- Map each claim stub from `CLAIMS_stub.md` into individual `C-XXX.md` pages.
- Backfill thesis excerpts and citations once we ingest `thesis.docx.md`.
- Wire CI to fail if any claim lacks notebook, data, or test anchors.
