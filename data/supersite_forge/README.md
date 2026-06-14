# Utah FORGE 16A / 16B — supersite data inventory (secondary, open EGS)

**Hot anchor for the regime diagram.** Utah FORGE is the only EGS
field laboratory in the global shortlist whose operator mandate
*is* open release through the DOE Geothermal Data Repository (GDR).
Every test cycle ships as a CC-BY DOI submission. We use FORGE as
the **high-Pe_T, high-Γ regime-diagram anchor** in this paper's
Figure 4: the regime-placement panel showing the three supersites
against the Huang-Pollack cloud.

- **Site:** Utah FORGE, Milford, Utah, USA
- **Coordinates:** 38.504 °N, −112.896 °W
- **Operator:** University of Utah Energy & Geoscience Institute
  under DOE GTO
- **Geometry:** doublet wells 16A(78)-32 (injector) and 16B(78)-32
  (producer), inclined into 175–225 °C granitoid at 2.1–3.3 km TVD
- **Why this site:** Class B1 in the supersite shortlist
  (19/20). The 2024 Extended Circulation Tests give a clean
  injector-producer pair to fit $\Gamma N_\alpha$ against.

## Datasets

All four datasets below are openly downloadable from the DOE GDR
(CC-BY). Operator-side download via the GDR API or the per-submission
landing pages.

### 1. GDR Submission 1149 — 16A baseline temperature logs

- **URL:** https://gdr.openei.org/submissions/1149
- **Variables:** static T-depth profiles in 16A(78)-32, June 2021
  baseline + post-stimulation logs
- **Format:** CSV / LAS
- **Time span:** 2021-06 to present
- **License:** CC-BY

### 2. GDR Submission 1326 — Phase 2C stimulation pressure

- **URL:** https://gdr.openei.org/submissions/1326
- **Variables:** wellhead pressure, flow rate, downhole pressure
  during the 58-32 phase-2C hydraulic stimulation
- **License:** CC-BY

### 3. GDR Submission 1608 — 2024 Extended Circulation Tests

- **URL:** https://gdr.openei.org/submissions/1608
- **Variables:** 30-second cadence T+p+flow at both 16A and 16B
  during the multi-week injector-producer circulation tests
- **License:** CC-BY
- **Role:** **primary dataset for the FORGE smoke forward**

### 4. GDR Submission 1683 — companion DTS / DAS arrays

- **URL:** https://gdr.openei.org/submissions/1683
- **Variables:** distributed-temperature-sensing fibre traces during
  the 2024 circulation tests
- **License:** CC-BY

## Role in the empirical paper

- **No joint Bayesian inversion.** The site is included as a
  **regime-diagram anchor** (Figure 4) and a **forward + ablation
  comparison** (Figure 5 panel). The 2-km depth and the hard-rock /
  fractured-granite physics are well outside the Cartesian
  1-D-column assumption that the Umiujaq paper otherwise uses.
- **Forward + ablation sweep.** Same five-point coupling sweep
  as Umiujaq, executed against the GDR 1608 circulation-test
  T+p record.
- **Regime placement.** Compute $\Gamma N_\alpha$, $\mathrm{Pe}_T$
  at the 16A injector depth using the GDR-published rock properties;
  show the FORGE point deep in the advection-dominated quadrant
  relative to the Huang-Pollack cloud median.
