# Auditor A2 — Implementation (Step-by-Step)

## 0) Scope enforcement
- Verify changed paths are limited to `docs/**`, `scaffold/**`, `mkdocs.yml`, `Makefile`.

## 1) Build docs
```bash
pip install mkdocs mkdocs-material
mkdocs build -q
```
- Fail if build errors.

## 2) Validate scaffolds
- Confirm `docs/<block>/index.md` exists for each block.
- Confirm `scaffold/<block>/README.md` present and describes write API.

## 3) Validate Makefile targets
- Ensure the listed targets exist (even as stubs).

## 4) Write review
- Use `ops/auditors/templates/PR_REVIEW_A2.md` to report findings.
