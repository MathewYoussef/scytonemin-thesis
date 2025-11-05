You are **Agent 3 — Data & Analysis Capture**.

**Start here:**
- Read your rules: `ops/agents/agent3_rules.md`
- Follow your step‑by‑step: `ops/agents/agent3_implementation.md`

**Goal:** Harvest datasets, models, and processed outputs from `Reflectance/`, `Initial_Calibration/`, `Act_of_God_Mamba_Results/`, and `Supplements/`. Introduce DVC/LFS, create sample datasets, export processed tables, and render figures into `scaffold/**/figures`.

**Inputs (read‑only):**
- `ops/output/inventory/inventory.csv` and `ops/output/proposals/mapping.csv`
- Scaffold paths from Agent 2 (`docs/**`, `scaffold/**`)

**Branch:** `agent3/data-harvest-setup`  
Open a **data-harvest** PR with catalog, dictionaries, samples, processed tables, and figures.
