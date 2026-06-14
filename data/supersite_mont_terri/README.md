# Mont Terri HE-D — supersite data inventory (secondary, digitized)

**Hero panel.** Mont Terri HE-D is the cleanest published demonstration
*anywhere* that the $\Gamma N_\alpha$ thermal-expansion term in the
mass equation cannot be set to zero: pore pressure rises from
~1 MPa to ~4 MPa as the surrounding Opalinus Clay heats from
15 °C to 100 °C, with a stable post-heating residual that decays on
the consolidation timescale. The site is treated as a **secondary**
target in this paper — forward + ablation runs only, no joint
Bayesian inversion — because the publicly-available time-series
data are at digitized-figure resolution rather than the full raw
archive.

- **Site:** Mont Terri Rock Laboratory, Saint-Ursanne, Switzerland
- **Coordinates:** 47.235 °N, 7.155 °E (HE-D heater at ~250-320 m
  below surface in the Opalinus Clay formation)
- **Operator:** swisstopo / Mont Terri Consortium (Nagra, BGR, ENSI,
  IRSN, JAEA, DOE-LBNL, et al.)
- **Geometry:** 1-D *radial* around a 5.4 m horizontal electric
  heater (HE-D campaign 2004-2005) — distinct from Umiujaq's 1-D
  *vertical* talik geometry. The coupled solver's
  `geometry="cylindrical"` mode (planned extension) is the natural
  target.
- **Why this site:** Class C1 in the supersite shortlist
  (18/20). The 1→4 MPa pressure rise is the textbook
  $\Gamma N_\alpha$ signal.

## Open-data path

The Mont Terri Consortium's raw archive requires consortium access
(out of scope for this paper per the no-negotiation constraint —
see the top-level plan). We use the **digitized published figures**
from Garitte et al. 2017 and Gens et al. 2007 as our authoritative
data source. The digitization is operator-prepared from:

| Source | Figure | Variables | Time span | Citation |
|---|---|---|---|---|
| Garitte et al. 2017, *J. Rock Mech. Geotech. Eng.* | Fig 8 | pore pressure at 4 boreholes near HE-D heater | 2004-04 to 2005-08 | doi:10.1016/j.jrmge.2017.07.001 |
| Gens et al. 2007, *Soils & Foundations* | Fig 12 | rock temperature at thermocouple grid | 2004-04 to 2005-08 | doi:10.3208/sandf.47.79 |
| Garitte et al. 2017, *J. Rock Mech. Geotech. Eng.* | Fig 5 (heater power schedule) | applied heater wattage W vs time | 2004-04 to 2005-08 | doi:10.1016/j.jrmge.2017.07.001 |

The digitization protocol uses `WebPlotDigitizer` (v4.7) with each
extracted CSV stored under `raw/` (gitignored). Operator notes for
the digitization step live in `raw/DIGITIZATION_NOTES.md` (to be
populated when the operator runs the digitization session). The
DOIs above are sufficient to track provenance back to the
peer-reviewed publication, not to the consortium archive.

## Out-of-scope / not used

- **HE-E (2011-present) and FE (2015-present) campaigns** — same
  consortium, same data-access constraint. Not used in this paper.
- **The full HE-D raw archive** — distributed on a consortium login,
  not openly downloadable. We deliberately do not request access.

## Role in the empirical paper

- **No joint inversion.** Published-figure resolution is too coarse
  (typically ~10 days between digitizable points across an 18-month
  campaign) for the adaptive-MH ensemble to constrain a five-parameter
  posterior. We do *not* attempt the joint inversion here.
- **Forward + ablation sweep.** With laboratory-grade material
  properties from Garitte 2017 (porosity, permeability, thermal
  conductivity all well-published for the Opalinus Clay), we drive
  the coupled solver across the same `gamma_n_alpha_scale ∈ {0,
  0.25, 0.5, 0.75, 1.0}` sweep used at Umiujaq.
- **Hero figure.** The Mont Terri panel in the manuscript's Figure 5
  shows the simulated pore-pressure response side-by-side with the
  Garitte 2017 digitized curve; the `s=0` baseline visibly cannot
  reproduce the 1→4 MPa rise.
