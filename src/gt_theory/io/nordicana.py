"""Readers for Nordicana D and Borealis CSV/TAB products covering the
Umiujaq Tasiapik valley supersite.

Three product families are supported, one entry point each:

- :func:`load_vdtbs_thermistors` — Borealis 10.5683/SP3/QSRW0I lithalsa
  thermistor bundle (multi-depth ground temperature at 5 boreholes on
  the VDTBS lithalsa mound, 2001-2024).
- :func:`load_immatsiak_head_series` — Nordicana D19
  10.5885/45309SL-15611D6EC6D34E23 drive-point piezometer hydraulic
  head time series (Immatsiak network, 2012-2023).
- :func:`load_d9_climate_series` — Nordicana D9
  10.5885/45120SL-067305A53E914AF0 climate-station scalar series (air
  T, snow depth, ground-surface T, etc.).

Each entry point returns an :class:`xarray.Dataset` on a tidy time (and
where applicable, depth) axis. Heavy regridding / unit conversion is
left to the supersite forward driver in `figures/empirical/`.

The parsers assume the Nordicana / Borealis convention of a leading
header block of `#`-commented lines followed by a delimited table. The
exact column names vary by product and are passed in by the caller;
the parser is generic over them so the same code path works for the
Borealis CSV bundle and the Nordicana per-variable zips.

Raw bytes live under `data/supersite_umiujaq/raw/` (gitignored).
Download recipe is in `data/supersite_umiujaq/README.md` — this loader
expects the operator to have already curl'd the zips.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

# Header lines in Nordicana/Borealis CSV bundles start with one of these
# comment markers.  pandas.read_csv with comment='#' handles the
# Nordicana style; Borealis sometimes uses '%' instead.
_COMMENT_MARKERS = ("#", "%")


def _read_csv_with_comments(
    path: str | Path,
    *,
    sep: str | None = None,
) -> pd.DataFrame:
    """Thin wrapper around :func:`pandas.read_csv` that tolerates both
    `#` and `%` header-comment markers and either comma or whitespace
    separators.

    Datetime parsing is deliberately deferred to the caller: pandas
    raises if a column passed via ``parse_dates`` is missing, which
    obscures the loader's own column-check error message.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Nordicana/Borealis file not found: {p}\n"
            f"Check the download recipe in data/supersite_umiujaq/README.md."
        )

    # Sniff the comment marker by reading the first non-blank line.
    comment = "#"
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            ls = line.lstrip()
            if not ls:
                continue
            if ls[:1] in _COMMENT_MARKERS:
                comment = ls[:1]
            break

    read_kwargs: dict[str, object] = {
        "comment": comment,
        "engine": "python",  # 'python' tolerates whitespace + delim quirks
    }
    if sep is not None:
        read_kwargs["sep"] = sep

    return pd.read_csv(p, **read_kwargs)


def _to_datetime(series: pd.Series, time_format: str | None) -> pd.DatetimeIndex:
    """Parse a column to ``pandas.Timestamp`` with an optional format."""
    return pd.to_datetime(series.to_numpy(), format=time_format)


# --------------------------------------------------------------------------
# Borealis VDTBS lithalsa thermistors — primary T-profile record
# --------------------------------------------------------------------------


def load_vdtbs_thermistors(
    path: str | Path,
    *,
    time_col: str = "datetime",
    depth_pattern: str = r"^T_(?P<depth>[0-9p\.]+)_m$",
    time_format: str | None = None,
) -> xr.Dataset:
    """Load one Borealis VDTBS thermistor CSV into a tidy
    `(time, depth)` xarray Dataset.

    Parameters
    ----------
    path
        Path to a single CSV from the Borealis bundle
        (doi:10.5683/SP3/QSRW0I).  Each file holds one borehole's
        thermistor record with one column per measurement depth.
    time_col
        Name of the datetime column in the CSV header.  The Borealis
        archive uses ``datetime``; override here if a later version
        renames it.
    depth_pattern
        Regex matched against every non-time column name to extract a
        depth in metres.  The default tolerates both ``T_0p5_m`` and
        ``T_0.5_m`` style headers (``p`` standing in for a decimal
        point).
    time_format
        Optional :meth:`pandas.to_datetime` format string.  Pass
        ``None`` to let pandas infer.

    Returns
    -------
    xarray.Dataset
        Variables: ``T_degC`` indexed by ``(time, depth_m)``.
        Coordinates carry the borehole filename as ``source_file``.

    Notes
    -----
    Confirm the exact column names against the bundled
    ``README.pdf`` / ``LISEZ-MOI.pdf`` inside the Borealis zip.
    """
    df = _read_csv_with_comments(path)

    if time_col not in df.columns:
        raise KeyError(
            f"Time column {time_col!r} not in {Path(path).name} (have: {list(df.columns)})"
        )

    pat = re.compile(depth_pattern)
    depth_cols: list[tuple[float, str]] = []
    for col in df.columns:
        if col == time_col:
            continue
        m = pat.match(str(col))
        if m is None:
            continue
        depth_str = m.group("depth").replace("p", ".")
        try:
            depth = float(depth_str)
        except ValueError:
            continue
        depth_cols.append((depth, col))

    if not depth_cols:
        raise ValueError(
            f"No thermistor depth columns matched {depth_pattern!r} in {Path(path).name}. "
            f"Inspect the header and pass an updated depth_pattern."
        )

    depth_cols.sort()
    depths = np.array([d for d, _ in depth_cols], dtype=float)
    values = np.stack([df[col].to_numpy(dtype=float) for _, col in depth_cols], axis=1)
    time = _to_datetime(df[time_col], time_format)

    return xr.Dataset(
        data_vars={
            "T_degC": (("time", "depth_m"), values),
        },
        coords={
            "time": time,
            "depth_m": depths,
        },
        attrs={
            "source_file": str(Path(path).resolve()),
            "source_doi": "10.5683/SP3/QSRW0I",
            "product": "Tasiapik VDTBS lithalsa thermistors",
        },
    )


# --------------------------------------------------------------------------
# Nordicana D19 — Immatsiak piezometer head series — primary p record
# --------------------------------------------------------------------------


def load_immatsiak_head_series(
    path: str | Path,
    *,
    time_col: str = "DateTime",
    value_col: str = "Head_m",
    station: str | None = None,
    time_format: str | None = None,
) -> xr.Dataset:
    """Load one Immatsiak drive-point piezometer head CSV into a
    `(time,)` xarray Dataset.

    Parameters
    ----------
    path
        Path to a single CSV from a Nordicana D19 per-variable zip.
    time_col, value_col
        Column names for the datetime and hydraulic-head columns.
        Defaults reflect the documented Nordicana D19 schema; override
        if your local extract uses different names.
    station
        Optional station label (``Immatsiak_1`` / ``Immatsiak_2`` /
        ``Immatsiak_3``) recorded into the Dataset attributes.
    time_format
        Optional :meth:`pandas.to_datetime` format string.

    Returns
    -------
    xarray.Dataset
        Variable: ``head_m`` indexed by ``time``.
    """
    df = _read_csv_with_comments(path)
    if time_col not in df.columns or value_col not in df.columns:
        raise KeyError(
            f"Expected columns {time_col!r}, {value_col!r} in {Path(path).name}; "
            f"have: {list(df.columns)}"
        )
    time = _to_datetime(df[time_col], time_format)
    head = df[value_col].to_numpy(dtype=float)

    attrs = {
        "source_file": str(Path(path).resolve()),
        "source_doi": "10.5885/45309SL-15611D6EC6D34E23",
        "product": "Immatsiak piezometer network",
    }
    if station is not None:
        attrs["station"] = station

    return xr.Dataset(
        data_vars={"head_m": (("time",), head)},
        coords={"time": time},
        attrs=attrs,
    )


# --------------------------------------------------------------------------
# Nordicana D9 — Umiujaq climate station — surface forcing
# --------------------------------------------------------------------------


def load_d9_climate_series(
    path: str | Path,
    *,
    time_col: str = "DateTime",
    value_col: str,
    variable_name: str,
    units: str,
    station: str | None = None,
    time_format: str | None = None,
) -> xr.Dataset:
    """Load one variable from a Nordicana D9 climate-station CSV into a
    `(time,)` xarray Dataset.

    The D9 product distributes one zip per variable per station.  Each
    extracted CSV has a single value column whose name encodes the
    physical quantity (``T_air_max_C``, ``Snow_depth_cm``,
    ``T_ground_surface_C``, ...).  This loader is intentionally
    generic; the caller specifies which column to lift and what to
    label it in the output.

    Parameters
    ----------
    path
        Path to one CSV from a Nordicana D9 per-variable zip.
    time_col
        Datetime column name.
    value_col
        Column to extract.
    variable_name
        Output variable name in the returned Dataset (e.g. ``T_air_degC``).
    units
        Units string recorded as a variable attribute.
    station
        Station label (e.g. ``VDTSYBU``, ``HT-234``).
    time_format
        Optional :meth:`pandas.to_datetime` format string.

    Returns
    -------
    xarray.Dataset
        Variable: ``<variable_name>`` indexed by ``time``.
    """
    df = _read_csv_with_comments(path)
    if time_col not in df.columns or value_col not in df.columns:
        raise KeyError(
            f"Expected columns {time_col!r}, {value_col!r} in {Path(path).name}; "
            f"have: {list(df.columns)}"
        )
    time = _to_datetime(df[time_col], time_format)
    values = df[value_col].to_numpy(dtype=float)

    attrs = {
        "source_file": str(Path(path).resolve()),
        "source_doi": "10.5885/45120SL-067305A53E914AF0",
        "product": "Nordicana D9 — Umiujaq climate station",
    }
    if station is not None:
        attrs["station"] = station

    ds = xr.Dataset(
        data_vars={variable_name: (("time",), values)},
        coords={"time": time},
        attrs=attrs,
    )
    ds[variable_name].attrs["units"] = units
    return ds


# --------------------------------------------------------------------------
# Cross-product alignment
# --------------------------------------------------------------------------


def align_to_hourly(
    *datasets: xr.Dataset,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    method: Literal["nearest", "pad", "ffill", "backfill", "bfill"] = "nearest",
    tolerance: pd.Timedelta | None = None,
) -> tuple[xr.Dataset, ...]:
    """Reindex each Dataset onto a common hourly time axis from *start*
    to *end* inclusive.

    Useful for stacking T-profile, piezometer head, and surface forcing
    into a single inversion-ready bundle without losing native sample
    spacing in the underlying products.
    """
    if tolerance is None:
        tolerance = pd.Timedelta("2h")
    target = pd.date_range(start=start, end=end, freq="1h")
    out = []
    for ds in datasets:
        out.append(ds.reindex(time=target, method=method, tolerance=tolerance))
    return tuple(out)
