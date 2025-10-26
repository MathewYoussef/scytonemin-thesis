# Scytonemin Thesis Audit Hub

Welcome to the shared launchpad for auditors, maintainers, and co-authors working across the Scytonemin thesis artifacts. This site describes how the blocks fit together, which repos or folders own each deliverable, and where evidence for every thesis claim will land once all agents finish their passes.

## How to Audit in 10 Minutes

1. **Scan the inventory** — Agent 1 surfaced current paths and proposed relocations in `ops/output/inventory/` and `ops/output/proposals/`. Use those CSVs to confirm inputs exist and to trace any moves once the restructure happens.
2. **Jump to a block** — Each section in the navigation outlines its analysis pipeline, required figures/tables, and the `scaffold/<block>/` destinations where Agents 3 & 4 will write artifacts.
3. **Verify the claims ledger** — The `Claims` section links every thesis statement to its notebook cell, processed dataset, and automated test once those integrations are wired.

## Responsibilities & Handoffs

- **Agent 2 (Scaffolding)** — Owns the structure under `docs/`, `scaffold/`, `mkdocs.yml`, and `Makefile`. All pages here are placeholders until the content agents populate them.
- **Agent 3 (Data & Analysis)** — Writes processed datasets, provenance, and generated visuals into the directories defined in this scaffold.
- **Agent 4 (Media)** — Curates photos/videos through `docs/media/**` and links them to block narratives.
- **Thesis manuscript** — `thesis.docx.md` (repo root) is the canonical source for prose excerpts. Use the headings in that file when filling each page’s “Thesis integration points” checklist.

## Integration Checklist

- `Makefile` exposes stubs for `make quickstart`, `make figures`, and `make audit`. Each target will call the corresponding pipelines once Agents 3 & 4 deliver notebooks and checks.
- `mkdocs build` should run cleanly at every commit; keep navigation aligned with the `scaffold/` layout so reviewers can trace evidence quickly.
