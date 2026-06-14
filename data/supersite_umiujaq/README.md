# Umiujaq (Tasiapik valley) — supersite data inventory

Primary supersite for the empirical coupled T–p paper. Tracks the four
openly-downloadable datasets that, together, provide the multi-decadal
co-located thermistor + drive-point piezometer + meteorological forcing
needed to invert ΓN_α in the silt-aquifer talik.

- **Site:** Tasiapik valley, Umiujaq, Nunavik, Québec, Canada
- **Coordinates:** ~56.55 °N, −76.55 °W (lithalsa core mound;
  UMIROCA borehole at 56.54213 °N, −76.52165 °W)
- **Operator:** Centre d'études nordiques (CEN, Université Laval) —
  Fortier / Lemieux / Allard / Sarrazin groups
- **Geometry:** 1-D vertical column through a thawing lithalsa and
  surrounding silt unit; supra-permafrost talik in direct hydraulic
  communication with the boreholes
- **Why this site:** Class A1 in the site-scoring rubric
  (17/20 score). Fortier et al. 2023 (*WRR*,
  doi:10.1029/2022WR032456) attribute ≈ half of mound thaw to
  advective heat from groundwater convergence — the exact regime
  where ΓN_α should bite.

## Datasets

All four datasets below are openly downloadable today; no registration
or PI-email negotiation required. Raw downloads live under `./raw/`
and are gitignored. Only this README, `site_config.yaml`, and any
small derived metadata files are tracked in git.

### 1. Tasiapik VDTBS lithalsa — multi-borehole thermistor record (primary T)

- **Title:** *Subsurface ground temperature data from an instrumented
  permafrost mound, Tasiapik Valley, Umiujaq, Nunavik, Québec,
  Canada (2001–2024)*
- **DOI:** `10.5683/SP3/QSRW0I` (Borealis / Scholars Portal Dataverse)
- **Landing:** https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/QSRW0I
- **Direct bulk-zip URL:**
  `https://borealisdata.ca/api/access/dataset/:persistentId/?persistentId=doi:10.5683/SP3/QSRW0I`
  → `doi-10.5683-sp3-qsrw0i_1.1.zip` (~9.4 MB)
- **Time span:** 28 Jun 2001 – 7 Sep 2024
- **Coverage:** 5 boreholes on/around the VDTBS lithalsa mound;
  thermistors at multiple depths near-surface to ~20 m. Exact depth
  list is in the bundled `README.pdf` / `LISEZ-MOI.pdf` (12 tabular
  files in the zip).
- **License:** **CC-BY 4.0**
- **Format:** CSV / TAB (also packaged as `.tab`-style 12.0 MB
  archival zip)
- **Citation:** Fortier, P., Fortier, R., Allard, M., Lemieux, J.-M.,
  Sarrazin, D. (2025). *Subsurface ground temperature data from an
  instrumented permafrost mound, Tasiapik Valley, Umiujaq, Nunavik,
  Québec, Canada (2001–2024).* Borealis, V1.
  https://doi.org/10.5683/SP3/QSRW0I
- **Role in this paper:** the canonical multi-decadal T-profile
  product. Supersedes the older partial Nordicana cuts cited in
  Lemieux 2016 (*J. Hydrol.*) and Fortier 2023 (*WRR*).

### 2. Nordicana D8 — UMIROCA single deep borehole (supplementary T)

- **Title:** *Borehole and near-surface ground temperatures in
  northeastern Canada* (UMIROCA station entry)
- **DOI:** `10.5885/45291SL-34F28A9491014AFD` (Nordicana D8, v. 1.6.0)
- **Landing:** http://nordicana.cen.ulaval.ca/en/publication?doi=45291SL-34F28A9491014AFD
- **Per-station bundle:** `ds_000633431.zip` on the D8 page
- **Time span at UMIROCA:** Oct 1997 – Oct 2023 (parent product
  1988–2023 across all sites)
- **Coverage:** single borehole at Umiujaq plateau; depths ≈ 11–27 m
  where specified; hourly / daily / monthly / yearly aggregations
  bundled.
- **License:** Nordicana D "Terms of use" (no explicit CC tag;
  attribution required, no registration wall)
- **Citation:** Allard, M., Sarrazin, D., L'Hérault, E. (2024).
  *Borehole and near-surface ground temperatures in northeastern
  Canada*, v. 1.6.0. Nordicana D8.
  https://doi.org/10.5885/45291SL-34F28A9491014AFD
- **Role in this paper:** plateau-side comparison column for the
  background (non-advective) thermal regime; not the primary
  inversion target.

### 3. Nordicana D19 — Immatsiak piezometer network (primary p)

- **Title:** *Immatsiak network of groundwater monitoring at
  Umiujaq in Nunavik, Quebec, Canada*
- **DOI:** `10.5885/45309SL-15611D6EC6D34E23` (Nordicana D19, v. 1.5)
- **Landing:** http://nordicana.cen.ulaval.ca/en/publication?doi=45309SL-15611D6EC6D34E23
- **Per-variable download IDs:** `id=4046` … `id=4054` at
  `http://www.nordicana.climatedata.ca/en/download?id=<n>` (one zip
  per variable × station: Immatsiak_1 / _2 / _3)
- **Time span:** Aug 2012 – Jun 2023
- **Coverage:** 9 drive-point piezometers across 7 valley sites; 3
  deep wells extending to 35 m at the instrumented sites
  (Immatsiak_1/_2/_3) with thermistor cables and solar-shielded
  air-T, all hourly.
- **License:** data ownership MELCC (Québec Ministry of
  Environment); released publicly through Nordicana D — attribution
  per Nordicana terms, no registration wall.
- **File sizes:** 1.46 – 17.57 MB per zip
- **Citation:** Fortier, R. (2026). *Immatsiak network of groundwater
  monitoring at Umiujaq in Nunavik, Quebec, Canada*, v. 1.5.
  Nordicana D19. https://doi.org/10.5885/45309SL-15611D6EC6D34E23
- **Role in this paper:** the pressure-time-series half of the joint
  T+p inversion. Co-located in the *same talik footprint* as the
  VDTBS thermistors (dataset 1).

### 4. Nordicana D9 — Umiujaq climate station (surface forcing)

- **Title:** *Climate station data from the Umiujaq region in
  Nunavik, Quebec, Canada*
- **DOI:** `10.5885/45120SL-067305A53E914AF0` (Nordicana D9,
  v. 1.10.0, updated 14 Feb 2026)
- **Landing:** http://nordicana.cen.ulaval.ca/en/publication?doi=45120SL-067305A53E914AF0
- **Per-variable download IDs:**
  - `id=4077` air Tmax
  - `id=4078` air Tmin
  - `id=4113` ground-T VDTSYBU
  - `id=4114` ground-T HT-234
  - `id=4115` ground-T HT-176
- **Time span:** 1997 – 2024 (rolling update)
- **Coverage:** air T, ground-surface T, RH, solar radiation, snow
  depth, wind speed/direction at the Umiujaq cluster (valley +
  plateau stations).
- **License:** Nordicana D "Terms of use", attribution required, no
  registration wall.
- **File sizes:** 0.28 – 14.33 MB per per-variable zip
- **Citation:** CEN (2026). *Climate station data from the Umiujaq
  region in Nunavik, Quebec, Canada*, v. 1.10.0. Nordicana D9.
  https://doi.org/10.5885/45120SL-067305A53E914AF0
- **Role in this paper:** GST history input to
  `run_column_coupled` and to the CRU-cross-check; snow / SAT
  series feeds the surface energy-balance term.

## Out-of-scope products (recorded so we don't chase them again)

- **Nordicana D46** (*Soil temperature and soil water content
  measurements near Umiujaq and in the Sheldrake catchment*,
  doi:10.5885/45567CE-639242EA518841D0). Surface-soil (0–15 cm) only,
  Umiujaq village + Sheldrake catchment — *not* the Tasiapik valley
  borehole product. Out of scope.
- No registered Nordicana entry matches the "CEN.SILA" shorthand
  used in some Lemieux/Fortier papers; what those papers call the
  SILA station record at Umiujaq is in practice the Nordicana D9
  product (dataset 4 above).

## Column-co-location map

Two columns are usable for the joint T+p inversion:

| Column | Thermistor source | Piezometer source | Notes |
|---|---|---|---|
| **Lithalsa mound (primary)** | dataset 1 (Borealis VDTBS) | dataset 3 (Nordicana D19, Immatsiak_1 or _2 — whichever is closest in plan view) | Best advective signal; Fortier 2023 attributes ~½ thaw to advection here. |
| **Plateau (secondary)** | dataset 2 (Nordicana D8 UMIROCA) | none | Background conduction-dominated reference; no co-located piezometer. |

The plateau column serves as the conduction-only control in the
ablation analysis; the primary inversion runs on the mound column.

## Download recipe (operator notes — run by hand, not in CI)

```bash
# 1. Tasiapik VDTBS thermistors (CC-BY)
curl -L -o raw/borealis-10.5683-SP3-QSRW0I.zip \
  "https://borealisdata.ca/api/access/dataset/:persistentId/?persistentId=doi:10.5683/SP3/QSRW0I"

# 2. Nordicana D8 UMIROCA — navigate the landing page to the
#    UMIROCA-specific bundle (file id may rotate between versions):
#    http://nordicana.cen.ulaval.ca/en/publication?doi=45291SL-34F28A9491014AFD
# (Manual download recommended; the page enumerates ds_000633431.zip
#  alongside other sites' bundles.)

# 3. Nordicana D19 Immatsiak piezometers — per-variable zips:
for id in 4046 4047 4048 4049 4050 4051 4052 4053 4054; do
  curl -L -o "raw/nordicana-D19-id${id}.zip" \
    "http://www.nordicana.climatedata.ca/en/download?id=${id}"
done

# 4. Nordicana D9 climate station — per-variable zips:
for id in 4077 4078 4113 4114 4115; do
  curl -L -o "raw/nordicana-D9-id${id}.zip" \
    "http://www.nordicana.climatedata.ca/en/download?id=${id}"
done
```

`raw/` is gitignored. `processed/` will hold the tidy xarray
Datasets emitted by `gt_theory.io.nordicana` once the loader is
implemented; those are also gitignored (see top-level `.gitignore`).

## Caveats

- Only the Borealis mirror (dataset 1) carries an explicit CC-BY 4.0
  tag. The three Nordicana D entries (D8 / D9 / D19) link to a
  generic Nordicana "Terms of use" page rather than a machine-readable
  CC tag; data are openly downloadable with attribution required.
- For D19, ownership belongs to MELCC; Nordicana is the distribution
  channel. Cite both MELCC and the Nordicana DOI in the manuscript.
- No PANGAEA or NSF Arctic Data Center mirror exists for D9 / D19.
  If Nordicana goes down during paper preparation, the Borealis
  mirror (dataset 1) is the only redundant copy and only covers
  thermistors.

## Cross-references

- Solver implementation: `src/gt_theory/solvers/column_coupled.py`
- Site config: `data/supersite_umiujaq/site_config.yaml`
- Loader: `src/gt_theory/io/nordicana.py` (planned)
- Forward-run outputs: `outputs/supersite_umiujaq/forward_runs.nc`
- Posterior samples (Phase 2b): `outputs/supersite_umiujaq/posterior_samples.npz`
