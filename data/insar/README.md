# InSAR cross-check ingest — open-data recipe

PS-InSAR (Persistent Scatterer Interferometric SAR) measures
millimetre-scale ground deformation from Sentinel-1 SLC time
series. We use it as an independent observable that probes the
coupled thermo-poroelastic strain field at every supersite, without
relying on borehole T or piezometric p.

## Primary source: European Ground Motion Service (EGMS)

- **License:** CC-BY 4.0 via the Copernicus Land Monitoring Service.
- **Coverage:** pan-European, 100 m grid, 2015-2024 (rolling).
- **Portal:** https://land.copernicus.eu/en/products/european-ground-motion-service
- **Products used in this paper:**
  - L3 Ortho-Vertical (VU): vertical displacement time series + linear velocity
  - L3 Ortho-East (EW): horizontal component (optional, second-pass)
- **Coverage status for our supersites:**
  - Mont Terri HE-D (47.235 °N, 7.155 °E): **EU-covered.** Use EGMS.
  - Umiujaq (56.55 °N, −76.55 °W) : **outside EU.** Use ASF DAAC fallback.
  - Utah FORGE (38.504 °N, −112.896 °W): **outside EU.** Use ASF DAAC fallback.

### Operator download recipe (Mont Terri)

```bash
# 1. Register at https://land.copernicus.eu/ (free).
# 2. Open the EGMS Explorer:
#    https://egms.land.copernicus.eu/
# 3. Pan / zoom to Mont Terri (lat 47.235, lon 7.155).
# 4. Tool > Download data.
#    - Product: "L3 Ortho-Vertical (VU)"
#    - Time window: 2015-01 to 2024-12
#    - Area of interest: 5 km radius around Mont Terri
#    - Format: CSV
# 5. Save as data/insar/raw/EGMS_L3_VU_mont_terri.csv
```

The resulting CSV is the format consumed by
``gt_theory.io.egms.load_egms_csv``.

### High-Pe_T H-P candidates inside EGMS coverage

The 13 H-P sites with `dT/dz > 50 K/km` (flagged red in
Figure~0c) that lie inside the EU domain are also candidates for an
InSAR cross-check. Identify them with:

```bash
PYTHONPATH=src python -c "
import pandas as pd
df = pd.read_parquet('outputs/global/ensemble_summary.parquet')
eu = df[(df.lat_deg >= 35) & (df.lat_deg <= 71) & (df.lon_deg >= -10) & (df.lon_deg <= 30)]
print(eu[eu.geo_gradient_K_per_km > 50][['site_id','lat_deg','lon_deg','geo_gradient_K_per_km']])
"
```

## Fallback for non-EU sites: ASF DAAC + HyP3

For Umiujaq and Utah FORGE, EGMS does not cover the bounding box.
The fallback path is:

1. Sentinel-1 SLC scenes from
   [ASF DAAC](https://search.asf.alaska.edu/) (NASA, free with
   Earthdata login).
2. Process with [HyP3](https://hyp3-docs.asf.alaska.edu/) -- a
   cloud-based ASF service that runs ISCE+MintPy and outputs PS
   time series.
3. The HyP3 product is also CSV-like; the same
   ``load_egms_csv`` reader applies after renaming the velocity
   column.

Operator note: HyP3 charges no money but credits and queueing
apply. Budget ~24 h for two processing jobs (Umiujaq, FORGE).

## What this directory holds

- `README.md` -- this file.
- `raw/` -- operator-downloaded CSVs (gitignored).
- `processed/` -- per-supersite reduced datasets ingested via
  ``gt_theory.io.egms.load_egms_csv``, written as netCDF
  (gitignored).
