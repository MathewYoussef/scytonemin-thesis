# Auditor A1 — Implementation (Step-by-Step)

## 0) Identify PR context
- Ensure PR has label `A1`.
- Checkout branch locally; set `BASE` to base ref and `HEAD` to head ref.

## 1) Path-scope enforcement
- List changed files: `git diff --name-only origin/${BASE}...origin/${HEAD}`
- Fail if any path is outside `ops/output/**` or `ops/logs/**`.

## 2) Artifact presence
- Confirm all expected files exist (see rules).

## 3) Validate inventory.csv
- Columns must include: `path,type,size_bytes,sha1,last_commit,language,category,block_guess,action_suggestion,new_location,notes`.
- Row count ≥ number of non-.git filesystem entries in `tree.txt` headerless list (±1 tolerance).

## 4) Validate large_files.csv and duplicates.csv
- Ensure outputs exist and are parsable; if empty, headers/stubs exist.

## 5) Check restructure_proposal.md & mapping.csv
- mapping.csv has the required headers and >= 1 entry for each non-trivial move.
- proposal contains target tree and rationale.

## 6) Write review
- Use `ops/auditors/templates/PR_REVIEW_A1.md` as your template and fill in status.
