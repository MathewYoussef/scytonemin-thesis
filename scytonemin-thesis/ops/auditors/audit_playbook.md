# Audit Playbook — Governance for Auditing Agents (A1–A4)

**Purpose:** Standardize auditing of Agents 1–4 via PR gating, clear path ownership, and objective checks. Auditors produce written reviews and CI passes must be green before merge.

## Labels and Triggers
- Agent 1 PRs → label `A1` → workflow `audit-agent1.yml`
- Agent 2 PRs → label `A2` → workflow `audit-agent2.yml`
- Agent 3 PRs → label `A3` → workflow `audit-agent3.yml`
- Agent 4 PRs → label `A4` → workflow `audit-agent4.yml`
- Workflow definitions are stored under `.github/workflows/audit-agent*.yml`; keep labels in sync with those filenames.

## Non‑interference
Auditors never modify source. They only:
- Add comments on PRs
- Commit reports under `ops/auditors/reports/` in separate reviewer branches (optional)
- Toggle labels/status per governance

## Merge Gates
- All step‑specific checks pass
- Scope enforcement passes (no cross‑path edits)
- Required artifacts present
- Auditor review status: **Approve**

## Escalation
If checks fail or scope violations occur:
- Mark PR with `needs-changes`
- Open issues referencing offending paths
- Re‑run audits on re‑push
