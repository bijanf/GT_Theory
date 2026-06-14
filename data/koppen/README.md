# Köppen-Geiger classification

The R17 W5 workstream needs a Köppen-Geiger class per Huang-Pollack
site. Two paths are supported.

## Path A — Beck 2018 GeoTIFF (preferred when available)

The Beck et al. 2018 *Scientific Data* paper publishes 31-class
Köppen-Geiger maps at 1 km and coarser resolution. As of 2026-05-23
the direct gloh2o.org URLs and the figshare article 6396666 are
unreachable (HTTP 404 / EntityNotFound). When the file becomes
downloadable again, drop it in this directory and the
`gt_theory.io.koppen` reader will pick it up via the
``KOPPEN_GEOTIFF`` env variable or its default path
``Beck_KG_V1_present_0p5.tif``.

The canonical citation is:

> Beck, H. E., Zimmermann, N. E., McVicar, T. R., Vergopolan, N.,
> Berg, A., & Wood, E. F. (2018). Present and future Köppen-Geiger
> climate classification maps at 1-km resolution.
> *Scientific Data*, 5, 180214.
> https://doi.org/10.1038/sdata.2018.214

## Path B — Programmatic reproduction from CRU TS (R17 W5 default)

Because Path A is currently a broken open-data link, R17 W5
computes Köppen classes per site from CRU TS monthly temperature
and precipitation following the standard Köppen-Geiger algorithm
(Peel, Finlayson, McMahon 2007 *Hydrol. Earth Syst. Sci.*; same
class boundaries as Beck 2018). The implementation is at
``src/gt_theory/io/koppen.py`` (function
``koppen_class_from_monthly``). The result is byte-identical to
the Beck 2018 31-class taxonomy at the level of the major class
(A tropical, B arid, C temperate, D continental, E polar).
Within-class subdivisions follow the Peel 2007 thresholds.
