# Scytonemin Thesis — GitHub Supplement Design & Crosswalk
_Last updated: 2025-10-26 07:44 UTC_

This document inventories every explicit **GitHub / Supplement** reference in the thesis and prescribes a **repo architecture** and **per‑section artifact plan** so reviewers can audit claims end‑to‑end.

---

## 1) External references detected in the thesis

**Counts**: GitHub mentions = **5** · Supplement/Appendix = **4** · DOI/External URLs = **109**

### 1.1 GitHub mentions (verbatim context snippets)
1. `Github` — -hour band-average reference powers were computed from the ten brightest midsummer days (15 Dec 2023–15 Jan 2024\) as: P‾₍MDV,UVA₎ \= 2.2671 mW·cm⁻²    and    P‾₍MDV,UVB₎ \= 0.0341 mW·cm⁻²(NOAA GML, n.d.; derivations on Github).  The SUV-100 system and companion radiometers at McMurdo are documented by NOAA; the site metadata place the station at 77°50′ S, 166°40′ E; 183 m ASL (NOAA GML, n.d.).    Dose definition. For grid g and band ∈
2. `Github` — 892032JYS2507117001, test date 2025-07-11). For each thallus we acquired fresh readings from both tiles and used their mean in normalization; the certificate’s wavelength-resolved reflectance (250–800 nm) is archived in Github.  #### Acquisition cadence  Integration time was fixed at 665 370 µs for all scans. For each orientation we recorded 150 spectra at 1-s cadence (600 spectra per thallus (4 orientations)). Primary analyses used the
3. `GitHub` — ra per thallus (4 orientations)). Primary analyses used the 12 o’clock and 6 o’clock orientations (aligned and counter-aligned with the exposure axis); the 3 o’clock and 9 o’clock blocks are retained as supplementary on GitHub and are not used in main-text statistics. A global dark set D() was computed as the mean of 250 dark frames acquired at session start. For each thallus, two fresh 6 % whites were recorded immediately prior to samp
4. `Github` — t training, the SSM was tested in relation to a held out 750-spectra panel constructed from three group IDs, a hard case from folds 01/05, a representative easy regime, and a zero-dose control (specifics are retained in Github); resulting in PSNR\_mean \= 44.153 dB, PSNR\_std \= 4.968 dB, SAM\_mean \= 1.039°. The panel governed early stopping and the ship/no-ship decision. Readiness gates required, on this panel: PSNR mean ≥ 18 dB, PSNR
5. `GitHub` — =0.060 mg·gDW⁻¹).  * **320–480 nm ROI:** Chromtotal​≈−6.1125×Occupancy+1.0763                  (r=−0.912,R²=0.831,RMSE=0.043 mg·gDW⁻¹).  Coefficient summaries for all models are provided in the supplementary material on GitHub (model summary), and the Σ, 320–480 nm fit is illustrated in Fig. 8.  ![][image12]  ## Discussion  Our results show that *Nostoc commune* responds to summer-like UV exposure with coordinated changes in pigments an

**Mapping placeholders (fill these):**
| Ref ID | Thesis excerpt (trimmed) | Intended repo | Planned path | Status |
|---|---|---|---|---|
| GH-001 | -hour band-average reference powers were computed from the ten brightest midsummer days (15 Dec 2023–15 Jan 2024\) as: P… | _(hub/analytics/data/instruments)_ | _(path/to/file)_ | TODO |
| GH-002 | 892032JYS2507117001, test date 2025-07-11). For each thallus we acquired fresh readings from both tiles and used their m… | _(hub/analytics/data/instruments)_ | _(path/to/file)_ | TODO |
| GH-003 | ra per thallus (4 orientations)). Primary analyses used the 12 o’clock and 6 o’clock orientations (aligned and counter-a… | _(hub/analytics/data/instruments)_ | _(path/to/file)_ | TODO |
| GH-004 | t training, the SSM was tested in relation to a held out 750-spectra panel constructed from three group IDs, a hard case… | _(hub/analytics/data/instruments)_ | _(path/to/file)_ | TODO |
| GH-005 | =0.060 mg·gDW⁻¹).  * **320–480 nm ROI:** Chromtotal​≈−6.1125×Occupancy+1.0763                  (r=−0.912,R²=0.831,RMSE=0… | _(hub/analytics/data/instruments)_ | _(path/to/file)_ | TODO |

### 1.2 Supplement / Appendix mentions (verbatim context snippets)
1. `Supplement` — UV varies strongly diurnally. From the same NOAA SUV-100 period, the 24-h coefficient of variation was \~0.48 for UV-A and \~0.79 for UV-B, with hourly UV-A spanning \~0.81–3.86 mW cm⁻² (calculations from NOAA data; see Supplement). Our chamber deliberately held UV-A constant (CV \= 0\) and confined UV-B to a short daily pulse (\~1.066 h) to isolate UV-specific pigment induction decoupled from PAR or diurnal gating (NOAA GML, n.d.).
2. `supplementary` — adence (600 spectra per thallus (4 orientations)). Primary analyses used the 12 o’clock and 6 o’clock orientations (aligned and counter-aligned with the exposure axis); the 3 o’clock and 9 o’clock blocks are retained as supplementary on GitHub and are not used in main-text statistics. A global dark set D() was computed as the mean of 250 dark frames acquired at session start. For each thallus, two fresh 6 % whites were recorded immediat
3. `si` — r noise without blurring narrow pigment features, we applied a one-dimensional sequence state-space model (Mamba-SSM) along the wavelength axis to each of the reflectance scan Rrelt,o,k().At wavelength i, a hidden state si​ is updated from si−1​ and the current reflectance xi​; selective gates decide how much new information to keep versus how much history to propagate, letting the model compare distant spectral regions. In contrast to
4. `supplementary` — r=-0.820,  R²=0.672,  RMSE=0.060 mg·gDW⁻¹).  * **320–480 nm ROI:** Chromtotal​≈−6.1125×Occupancy+1.0763                  (r=−0.912,R²=0.831,RMSE=0.043 mg·gDW⁻¹).  Coefficient summaries for all models are provided in the supplementary material on GitHub (model summary), and the Σ, 320–480 nm fit is illustrated in Fig. 8.  ![][image12]  ## Discussion  Our results show that *Nostoc commune* responds to summer-like UV exposure with coordina

### 1.3 Other external links
- 109 DOI/URL references detected. For auditability, cite the **GitHub release DOI** (Zenodo) for this thesis hub in your *Code & Data Availability* section.

---

## 2) GitHub architecture (repos, responsibilities, and reviewer UX)

**Goal:** a 10‑minute path to reproduce main figures on a laptop, and a full raw→results pipeline via containerized runs. Each claim is traceable to code, data, and tests.

### 2.1 Repos
1. **`scytonemin-thesis`** *(Hub & Audit Portal)* — docs site, claims index, light notebooks for figures, CI to rebuild on sample data.
2. **`scytonemin-analytics`** *(Code & Models)* — reflectance pipeline, Mamba SSM denoising (training & inference), chromatography analysis, stats, figure builders.
3. **`scytonemin-data`** *(Data & Metadata)* — sample data in repo; full datasets via **DVC**/**Git LFS** to a remote (Zenodo/OSF/S3).
4. **`scytonemin-instruments`** *(Hardware & Calibrations)* — SUV‑100 geometry, light programs, radiometric calibrations, environmental logs, SOPs.

> Prefer the 4‑repo pattern (clean separation). A monorepo is acceptable but must preserve the same top‑level directories and CI targets.

### 2.2 Mandatory features
- **Releases w/ DOI**: Tag `v1.0.0` at submission; archive to Zenodo; cite DOI + commit hash in the thesis.
- **Environment reproducibility**: `environment.yml` and `Dockerfile`; deterministic seeds; CPU‑only fallbacks.
- **CI**: GitHub Actions that run `make audit` on the hub (executing all figure notebooks on sample data), plus unit tests in `scytonemin-analytics`.
- **Data governance**: sample data for quick checks; full data pulled with `dvc pull`; all raw & derived files carry `sha256` checksums and a **data dictionary**.
- **Documentation**: MkDocs site (Material theme) in the hub with pages per claim and **How to Audit in 10 Minutes**.

---

## 3) Candidate claims to register in `CLAIMS.md`
Below are automatically extracted candidate claims (heuristic). Curate into `C-###` entries, each linking to a notebook cell, code path, data file, and a test.

- **C-001:** Beyond Earth, optical remote sensing (ORS) has been instrumental in mapping Martian mineralogy, sedimentary deposits,  geological structures, and paleo-aqueous environments (Murchie et al., 2007; Poulet et al., 2005; Ehlmann et al., 2008).
- **C-002:** In Antarctica, *Nostoc*\-dominated mats are widespread in the wetted zones of the McMurdo Dry Valleys (MDV), where they play a significant role in primary production and biogeochemical cycling during brief seasonal melt periods, such as those associated with wetted moats, slow-flowing streams, and snow-edge habitats.
- **C-003:** Variation in UV dose among treatments was achieved by positioning thalli within pre-mapped columns of an irradiance grid, which provided spatially distinct exposure intensities (see *Dosimetry* for mapping methodology and percent-MDV calculations).
- **C-004:** The mapping instrument was a Solarmeter Model 5.0 (Total UV A+B), specified for 280–400 nm with 0–199.9 mW cm⁻² range and ±5 % accuracy (Solarmeter, n.d.; Forestry-Suppliers, n.d.).
- **C-005:** Under the measured powers and these lamp times, Grid 3 delivered \~100 % MDV UV-B and \~110 % MDV UV-A by design; full mapping appears in *table 1*.
- **C-006:** In contrast to earlier SSMs (e.g., S4), Mamba makes the state‑space parameters input‑dependent (“selective”), which improves modeling of information‑dense sequences.
- **C-007:** The model predicts a denoised estimate Ṙ from input *Rrel*, while DipAwareLoss refits a guarded baseline to both the prediction and target, subtracts it to obtain absorption depth profiles, and penalizes the resulting area, centroid, and depth mismatches, thereby steering the network to preserve those features under the locally fitted continuum.
- **C-008:** |
| Gate 4 — Downstream proxy | **PASS** | Replicate variance: 2.92×10⁻⁵ → 1.38×10⁻⁵ (52.6% reduction), Separability ratio: 21.30 → 30.34 (Δ \= 9.04),  Dose monotonicity: intact (median Δρ \= 0, improved ratio \= 1.0).
- **C-009:** |

##### Variance and “dip behavior” (signs of under- or over-fitting)

Because the model is trained in a noise-to-noise setting, its output is effectively the average clean signal consistent with a given noisy measurement and its metadata.
- **C-010:** Nevertheless, for the purposes of this thesis, it is adequate, because: (i) coverage‑style tail reliability is high; (ii) downstream separability and dose monotonicity are preserved; and (iii) fundamentally, the increase in SNR provides a foundational spectra from which analytics can be based upon (*figure 4*).
- **C-011:** Colors show the change in robust signal-to-noise ratio (SNR), expressed in decibels using a median/MAD estimator; red tones indicate SNR gains, blue tones indicate losses.
- **C-012:** Right-margin values report each treatment’s mean ΔSNR inside the ROI (320-480), and rows are ordered by those in-band gains so higher improvements appear at the bottom.

> **Action:** For each curated claim, add: (a) thesis citation (section/figure), (b) figure notebook path, (c) input datasets (file + checksum), (d) code refs (repo@commit:path), (e) tests & thresholds.

---

## 4) Section‑by‑section GitHub artifact plan
For each thesis section, place the following artifacts so a reviewer can validate parameters, inputs, and outputs.

#### Quantification of the Scytonemin Fingerprint with Proximal Reflectance Spectra

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Abstract

**Recommended GitHub artifacts and design:**
- High-level **CLAIMS.md** references (no data), and link to DOI-tagged release.

#### Introduction

**Recommended GitHub artifacts and design:**
- Background citations only; no repo content beyond a brief **project overview** in the hub README.

#### Materials and Methods

**Recommended GitHub artifacts and design:**
- Create **/docs/methods/** pages with SOPs and parameter tables; link to code and data sources.

#### Organism and rationale

**Recommended GitHub artifacts and design:**
- SOPs for field handling; **metadata schema** (CSV/JSON) and example filled rows; de-identify locations if needed.

#### Field collection and storage

**Recommended GitHub artifacts and design:**
- SOPs for field handling; **metadata schema** (CSV/JSON) and example filled rows; de-identify locations if needed.

#### Experimental design

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Experimental unit, treatment structure, and layout

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Dosimetry (UV-A & UV-B dose quantification)

**Recommended GitHub artifacts and design:**
- Provide **light-program CSVs**, **sensor calibration** files, and a notebook that recomputes doses ± tolerance; unit tests asserting dose bounds.

#### Rehydration and PAR revival

**Recommended GitHub artifacts and design:**
- SOP and logs for hydration cycles; include time-stamped records used to exclude/flag scans.

#### Light program, calibration, and conditions

**Recommended GitHub artifacts and design:**
- Provide **light-program CSVs**, **sensor calibration** files, and a notebook that recomputes doses ± tolerance; unit tests asserting dose bounds.

#### Hydration management (optical confound control)

**Recommended GitHub artifacts and design:**
- SOP and logs for hydration cycles; include time-stamped records used to exclude/flag scans.

#### Enclosure and environmental readbacks

**Recommended GitHub artifacts and design:**
- Enclosure diagrams/photos; **environmental logs** (temp/humidity) and parsing code.

#### White references and per‑sample normalization.

**Recommended GitHub artifacts and design:**
- White-tile characterization, **drift tracking**, and normalization code; test comparing pre/post response.

#### Reflectance spectroscopy

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Instrument and geometry

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Acquisition cadence

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Computation, normalization, & denoising

**Recommended GitHub artifacts and design:**
- White-tile characterization, **drift tracking**, and normalization code; test comparing pre/post response.
- Model code (Mamba SSM), **training configs**, **weights artifacts**, evaluation notebooks and ablations; seed control for determinism.

#### Per-scan reflectance (ratio-of-radiances)

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Denoising (Mamba SSM) at the single-scan level

**Recommended GitHub artifacts and design:**
- Model code (Mamba SSM), **training configs**, **weights artifacts**, evaluation notebooks and ablations; seed control for determinism.

#### Pigment extraction for UPLC

**Recommended GitHub artifacts and design:**
- Raw vendor exports or instructions; **standards curves**, **peak-integration scripts**, and **weighting rationale** with tests on a standard run.

#### Freeze‑drying and storage

**Recommended GitHub artifacts and design:**
- SOPs for field handling; **metadata schema** (CSV/JSON) and example filled rows; de-identify locations if needed.

#### Extraction protocol (scytonemin‑optimized)

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Evaporation and reconstitution

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### UPLC–DAD–MS

**Recommended GitHub artifacts and design:**
- Raw vendor exports or instructions; **standards curves**, **peak-integration scripts**, and **weighting rationale** with tests on a standard run.

#### Instrumentation

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Chromatographic conditions

**Recommended GitHub artifacts and design:**
- Raw vendor exports or instructions; **standards curves**, **peak-integration scripts**, and **weighting rationale** with tests on a standard run.

#### Mass spectrometry

**Recommended GitHub artifacts and design:**
- Raw vendor exports or instructions; **standards curves**, **peak-integration scripts**, and **weighting rationale** with tests on a standard run.

#### Analytical standards and calibration

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Fitting strategy and weighting

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Operational bounds and reporting

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Propagation to dry‑weight metrics and pipeline concordance

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Results

**Recommended GitHub artifacts and design:**
- Figure notebooks (CPU-friendly) using **sample datasets**; store generated images in `/figures/` via CI.

#### Robust Mean Concentration and Dose

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Concentration profiles across the UVA–UVB dose series

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Cross‑assay agreement (Chrom vs. DAD)

**Recommended GitHub artifacts and design:**
- Raw vendor exports or instructions; **standards curves**, **peak-integration scripts**, and **weighting rationale** with tests on a standard run.
- Notebook computing correlation and CI; test asserting r ≥ threshold and matching reported value ± tolerance.

#### Synthesis of dose–response pattern

**Recommended GitHub artifacts and design:**
- Dose–response modeling notebook; bootstrap CIs; pre-specified model and alternatives.

#### Data handling and artifacts

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Reflectance Spectra (Dose and Concentration)

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Dose effects on the occupancy metric

**Recommended GitHub artifacts and design:**
- Metric implementation and **inverse relationship** check vs. concentration; include a test for monotonicity.

#### ![][image10]

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### #### Occupancy is inversely related to scytonemin concentration

**Recommended GitHub artifacts and design:**
- Metric implementation and **inverse relationship** check vs. concentration; include a test for monotonicity.

#### Occupancy and dose track the inverse of concentration

**Recommended GitHub artifacts and design:**
- Metric implementation and **inverse relationship** check vs. concentration; include a test for monotonicity.

#### From reflectance to concentration: empirical mapping

**Recommended GitHub artifacts and design:**
- End‑to‑end notebook from spectra → concentration with **error propagation**; archive model parameters.

#### Discussion

**Recommended GitHub artifacts and design:**
- Link every sentence-level claim to **CLAIMS.md** entries; no new data here.

#### Instrument‑specific influences (sensor, angle, and geometry)

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Noise in the data, temperature control, and light source

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Chromatographic quantification and matrix effects (HPLC‑DAD/PDA)

**Recommended GitHub artifacts and design:**
- Raw vendor exports or instructions; **standards curves**, **peak-integration scripts**, and **weighting rationale** with tests on a standard run.

#### Chlorophyll and β‑carotene were not quantified (implications for ratios)

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Redox forms and spectral fingerprints

**Recommended GitHub artifacts and design:**
- Raw vendor exports or instructions; **standards curves**, **peak-integration scripts**, and **weighting rationale** with tests on a standard run.

#### Environmental dose benchmarking and site caveats

**Recommended GitHub artifacts and design:**
- Enclosure diagrams/photos; **environmental logs** (temp/humidity) and parsing code.

#### Reconciling reflectance and concentration: the “inversion”

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Replication and statistical power

**Recommended GitHub artifacts and design:**
- Scripts to recompute robust means, CIs, and power; store **simulation seeds** and results tables.

#### Hypothesis appraisal and Conclusion

**Recommended GitHub artifacts and design:**
- Link every sentence-level claim to **CLAIMS.md** entries; no new data here.

#### Acknowledgements

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.

#### Bibliography:

**Recommended GitHub artifacts and design:**
- Cross-link to relevant code/data; add a short **SECTION.md** explaining artifacts and reproducibility hooks.


---

## 5) Hub repo skeleton (paste‑ready)

```
scytonemin-thesis/
├─ README.md
├─ CLAIMS.md
├─ claims/
│  ├─ C-001.md
│  └─ ...
├─ manifest/claims.yml
├─ notebooks/
│  ├─ fig01_dose_response.ipynb
│  ├─ fig02_occupancy_vs_concentration.ipynb
│  └─ ...
├─ figures/                 # generated by CI
├─ env/
│  ├─ environment.yml
│  ├─ requirements.txt
│  └─ Dockerfile
├─ Makefile                 # make quickstart | figures | audit
├─ .github/workflows/ci-audit.yml
├─ CITATION.cff
├─ LICENSE
└─ docs/                    # MkDocs (GitHub Pages)
```

**`CLAIMS.md` example row**
```
| ID    | Claim (thesis text)                                         | Evidence                               |
|-------|--------------------------------------------------------------|----------------------------------------|
| C-001 | Occupancy is inversely related to scytonemin concentration.  | nb: notebooks/fig02_occupancy...       |
|       |                                                              | data: scytonemin-data/.../occupancy.csv|
|       |                                                              | code: scytonemin-analytics/.../metrics |
|       |                                                              | test: tests/test_occupancy_mapping.py  |
```

---

## 6) CI tests that *prevent drift*
- **Figure parity tests**: assert summary statistics (e.g., correlation r, slope, CI) hit expected ranges.
- **Dose recomputation**: recompute UV‑A/UV‑B doses from logs; fail if outside ± tolerance.
- **Chromatography regression**: re‑integrate a standard run; assert concentrations within tolerance.
- **Denoising determinism**: same seed → same outputs; ablations documented.- **Data integrity**: checksums for all critical inputs; schema validation for metadata.

---

## 7) Action checklist
1. Create the four repos; enable Zenodo and GitHub Pages for the hub.
2. Add environment files, Makefile, CI workflows, LICENSE, CITATION.
3. Populate **sample data** + data dictionary; wire `make quickstart` to generate Fig 1–3.
4. Extract the five GitHub‑referenced derivations/materials—map each to the table above and pin them to a tag.
5. Curate 8–12 key claims into `CLAIMS.md` and wire each to a notebook, code path, and dataset.
6. Tag `v0.1.0` to dry‑run CI and the docs site; then freeze `v1.0.0` for submission and cite DOI + commit in the thesis.

---

## 8) Boilerplate for the thesis (“Code & Data Availability”)

> **Code & Data Availability.** All analysis code, models, and documentation are archived at **Scytonemin Thesis — Audit Hub** (Release **v1.0.0**; DOI: *[insert DOI]*). The hub reproduces every figure on a sample dataset (`make quickstart`) and verifies each claim against versioned inputs. Full raw and processed datasets are managed with DVC and retrievable via `make data`. Exact software versions and commit hashes are pinned (primary analysis at commit **`<hash>`**). Instrument geometry, calibrations, dosimetry schedules, and SOPs are provided in the **Instruments** repository.

---

### Appendix A — Full list of GitHub & Supplement excerpts (for traceability)

**GitHub excerpts**  
1. `Github` — -hour band-average reference powers were computed from the ten brightest midsummer days (15 Dec 2023–15 Jan 2024\) as: P‾₍MDV,UVA₎ \= 2.2671 mW·cm⁻²    and    P‾₍MDV,UVB₎ \= 0.0341 mW·cm⁻²(NOAA GML, n.d.; derivations on Github).  The SUV-100 system and companion radiometers at McMurdo are documented by NOAA; the site metadata place the station at 77°50′ S, 166°40′ E; 183 m ASL (NOAA GML, n.d.).    Dose definition. For grid g and band ∈
2. `Github` — 892032JYS2507117001, test date 2025-07-11). For each thallus we acquired fresh readings from both tiles and used their mean in normalization; the certificate’s wavelength-resolved reflectance (250–800 nm) is archived in Github.  #### Acquisition cadence  Integration time was fixed at 665 370 µs for all scans. For each orientation we recorded 150 spectra at 1-s cadence (600 spectra per thallus (4 orientations)). Primary analyses used the
3. `GitHub` — ra per thallus (4 orientations)). Primary analyses used the 12 o’clock and 6 o’clock orientations (aligned and counter-aligned with the exposure axis); the 3 o’clock and 9 o’clock blocks are retained as supplementary on GitHub and are not used in main-text statistics. A global dark set D() was computed as the mean of 250 dark frames acquired at session start. For each thallus, two fresh 6 % whites were recorded immediately prior to samp
4. `Github` — t training, the SSM was tested in relation to a held out 750-spectra panel constructed from three group IDs, a hard case from folds 01/05, a representative easy regime, and a zero-dose control (specifics are retained in Github); resulting in PSNR\_mean \= 44.153 dB, PSNR\_std \= 4.968 dB, SAM\_mean \= 1.039°. The panel governed early stopping and the ship/no-ship decision. Readiness gates required, on this panel: PSNR mean ≥ 18 dB, PSNR
5. `GitHub` — =0.060 mg·gDW⁻¹).  * **320–480 nm ROI:** Chromtotal​≈−6.1125×Occupancy+1.0763                  (r=−0.912,R²=0.831,RMSE=0.043 mg·gDW⁻¹).  Coefficient summaries for all models are provided in the supplementary material on GitHub (model summary), and the Σ, 320–480 nm fit is illustrated in Fig. 8.  ![][image12]  ## Discussion  Our results show that *Nostoc commune* responds to summer-like UV exposure with coordinated changes in pigments an

**Supplement/Appendix excerpts**  
1. `Supplement` — UV varies strongly diurnally. From the same NOAA SUV-100 period, the 24-h coefficient of variation was \~0.48 for UV-A and \~0.79 for UV-B, with hourly UV-A spanning \~0.81–3.86 mW cm⁻² (calculations from NOAA data; see Supplement). Our chamber deliberately held UV-A constant (CV \= 0\) and confined UV-B to a short daily pulse (\~1.066 h) to isolate UV-specific pigment induction decoupled from PAR or diurnal gating (NOAA GML, n.d.).
2. `supplementary` — adence (600 spectra per thallus (4 orientations)). Primary analyses used the 12 o’clock and 6 o’clock orientations (aligned and counter-aligned with the exposure axis); the 3 o’clock and 9 o’clock blocks are retained as supplementary on GitHub and are not used in main-text statistics. A global dark set D() was computed as the mean of 250 dark frames acquired at session start. For each thallus, two fresh 6 % whites were recorded immediat
3. `si` — r noise without blurring narrow pigment features, we applied a one-dimensional sequence state-space model (Mamba-SSM) along the wavelength axis to each of the reflectance scan Rrelt,o,k().At wavelength i, a hidden state si​ is updated from si−1​ and the current reflectance xi​; selective gates decide how much new information to keep versus how much history to propagate, letting the model compare distant spectral regions. In contrast to
4. `supplementary` — r=-0.820,  R²=0.672,  RMSE=0.060 mg·gDW⁻¹).  * **320–480 nm ROI:** Chromtotal​≈−6.1125×Occupancy+1.0763                  (r=−0.912,R²=0.831,RMSE=0.043 mg·gDW⁻¹).  Coefficient summaries for all models are provided in the supplementary material on GitHub (model summary), and the Σ, 320–480 nm fit is illustrated in Fig. 8.  ![][image12]  ## Discussion  Our results show that *Nostoc commune* responds to summer-like UV exposure with coordina
