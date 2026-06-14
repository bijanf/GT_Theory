"""Reader for the NOAA Paleoclimatology / Huang-Pollack 2000-2013 borehole
temperature archive.

Each archive file is a plain-text borehole log with a long ``#``-commented
header carrying site metadata, followed by a whitespace-separated table of
depth (m) vs. temperature (deg C).  This module parses both parts into a
typed :class:`BoreholeProfile`.

Source archive: https://www.ncei.noaa.gov/access/paleo-search/study/1000889
File naming convention: ``huang-YYYY-<CC>-<site>.txt`` where ``CC`` is the
ISO country code (or ``AL-Kolonja`` style for some sites).

Example
-------
>>> from gt_theory.io import parse_huang_pollack
>>> profile = parse_huang_pollack("tests/data/huang-2013-AU-10.txt")
>>> profile.site_id, profile.lat_deg, profile.depth_m.size
('AU-10', -34.0, 57)
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Lines that introduce the tabular data block; on the next non-blank line
# the numeric data begins.
_DATA_HEADER_RE = re.compile(r"^\s*Depth_m\s+Temperature_Celsius", re.IGNORECASE)


@dataclass(frozen=True)
class BoreholeProfile:
    """One borehole log + its site metadata.

    Attributes
    ----------
    site_id
        Short site identifier, e.g. ``"AU-10"``.
    country
        Country name as recorded in the archive header.
    lat_deg, lon_deg
        Site coordinates in decimal degrees.
    max_depth_m
        Maximum logged depth from the archive header (m).
    measurement_year
        Decimal year of measurement, if recorded; otherwise None.
    prior_gst_c
        Estimated prior steady-state ground-surface temperature (deg C).
    mean_conductivity_w_m_k
        Estimated mean thermal conductivity (W m^-1 K^-1).
    mean_gradient_k_per_km
        Estimated mean thermal gradient (K km^-1).
    depth_m
        Logged depths in metres, shape ``(n,)``, sorted ascending.
    temperature_c
        Logged temperatures in deg C, shape ``(n,)``.
    source_path
        Absolute path of the parsed file.
    """

    site_id: str
    country: str
    lat_deg: float
    lon_deg: float
    max_depth_m: float
    measurement_year: float | None
    prior_gst_c: float | None
    mean_conductivity_w_m_k: float | None
    mean_gradient_k_per_km: float | None
    depth_m: np.ndarray
    temperature_c: np.ndarray
    source_path: Path

    def __post_init__(self) -> None:
        if self.depth_m.shape != self.temperature_c.shape:
            raise ValueError(
                f"depth_m {self.depth_m.shape} vs temperature_c {self.temperature_c.shape}"
            )
        if self.depth_m.ndim != 1:
            raise ValueError("depth_m must be 1-D.")
        if np.any(np.diff(self.depth_m) <= 0):
            raise ValueError("depth_m must be strictly increasing.")


def _strip_value(raw: str) -> str:
    """Strip a header value string of trailing units in parens and units like 'm'.

    e.g. ``"449.580 m"`` -> ``"449.580"``; ``"Australia"`` -> ``"Australia"``.
    """
    text = raw.strip()
    # Trim trailing unit words separated by whitespace.
    parts = text.split()
    if len(parts) >= 2 and parts[-1].lower() in {"m", "km", "k/km", "w/m/k"}:
        text = " ".join(parts[:-1])
    return text.strip()


def _maybe_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(_strip_value(raw))
    except (TypeError, ValueError):
        return None


def _parse_header(lines: list[str]) -> dict[str, str]:
    """Pull ``# key: value`` (or ``# key (units): value``) pairs from header
    lines.  Returns a dict keyed by normalised (lowercase, underscore-joined)
    keys."""
    out: dict[str, str] = {}
    pat = re.compile(r"^\s*#\s*([^:#]+?)\s*:\s*(.*\S)\s*$")
    for line in lines:
        m = pat.match(line)
        if not m:
            continue
        key_raw, val_raw = m.group(1), m.group(2)
        key = re.sub(r"\s*\([^)]*\)\s*", "", key_raw).strip()
        key = re.sub(r"[^\w]+", "_", key).strip("_").lower()
        if key and key not in out:
            out[key] = val_raw.strip()
    return out


def parse_huang_pollack(path: str | Path) -> BoreholeProfile:
    """Parse one Huang-Pollack archive file into a :class:`BoreholeProfile`."""
    p = Path(path).expanduser().resolve()
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Split into header lines (any line beginning with '#') and the data
    # block.  The data block starts after a line whose contents are the
    # ``Depth_m  Temperature_Celsius  Notes`` header.
    data_start: int | None = None
    for i, line in enumerate(lines):
        if _DATA_HEADER_RE.match(line):
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError(f"No data header found in {p}")

    header_lines = [ln for ln in lines[:data_start] if ln.lstrip().startswith("#")]
    body_lines = lines[data_start:]

    meta = _parse_header(header_lines)

    site_id = meta.get("site_name") or p.stem.replace("huang-2013-", "")
    country = meta.get("country", "")
    lat = _maybe_float(meta.get("northernmost_latitude"))
    if lat is None:
        lat = _maybe_float(meta.get("southernmost_latitude"))
    lon = _maybe_float(meta.get("easternmost_longitude"))
    if lon is None:
        lon = _maybe_float(meta.get("westernmost_longitude"))
    if lat is None or lon is None:
        raise ValueError(f"{p}: missing lat/lon in header")

    max_depth_m = _maybe_float(meta.get("maximum_depth")) or 0.0

    # Numeric metadata; the archive uses keys like
    # ``date_of_measurement_year`` because of the ``(year)`` suffix.
    measurement_year = _maybe_float(meta.get("date_of_measurement"))
    prior_gst_c = _maybe_float(meta.get("estimated_prior_steady_state_gst"))
    mean_kappa_w_m_k = _maybe_float(meta.get("estimated_mean_conductivity"))
    mean_grad_k_per_km = _maybe_float(meta.get("estimated_mean_thermal_gradient"))

    # Parse the data block.
    depths: list[float] = []
    temps: list[float] = []
    for raw in body_lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            d = float(parts[0])
            t = float(parts[1])
        except ValueError:
            continue
        depths.append(d)
        temps.append(t)

    if len(depths) < 5:
        raise ValueError(f"{p}: only {len(depths)} data rows found; need >= 5")

    order = np.argsort(depths)
    depth_arr = np.array(depths, dtype=float)[order]
    temp_arr = np.array(temps, dtype=float)[order]
    # Drop duplicate depths if any (keep the first).
    keep = np.concatenate([[True], np.diff(depth_arr) > 0])
    depth_arr = depth_arr[keep]
    temp_arr = temp_arr[keep]

    return BoreholeProfile(
        site_id=site_id,
        country=country,
        lat_deg=float(lat),
        lon_deg=float(lon),
        max_depth_m=float(max_depth_m or depth_arr[-1]),
        measurement_year=measurement_year,
        prior_gst_c=prior_gst_c,
        mean_conductivity_w_m_k=mean_kappa_w_m_k,
        mean_gradient_k_per_km=mean_grad_k_per_km,
        depth_m=depth_arr,
        temperature_c=temp_arr,
        source_path=p,
    )


def iter_borehole_archive(
    directory: str | Path,
    *,
    pattern: str = "huang-*.txt",
    strict: bool = False,
) -> Iterator[BoreholeProfile]:
    """Yield :class:`BoreholeProfile` for every Huang-Pollack file under
    *directory*.  Unparseable files are skipped (with a warning printed) when
    ``strict`` is False, or re-raised when ``strict`` is True."""
    import sys

    root = Path(directory).expanduser().resolve()
    for path in sorted(root.glob(pattern)):
        try:
            yield parse_huang_pollack(path)
        except Exception as exc:
            if strict:
                raise
            print(f"WARN: failed to parse {path.name}: {exc}", file=sys.stderr)


def find_borehole_file(site_id: str, data_root: str | Path) -> Path:
    """Resolve a Huang-Pollack file path from its ``Site_Name`` (e.g.
    ``"AU-10"``) by searching *data_root* recursively for
    ``huang-*-{site_id}.txt``.
    """
    root = Path(data_root).expanduser().resolve()
    matches = list(root.rglob(f"huang-*-{site_id}.txt"))
    if not matches:
        raise FileNotFoundError(f"No Huang-Pollack file for site_id={site_id!r} under {root}")
    if len(matches) > 1:
        # Keep the lexicographically first to be deterministic.
        matches.sort()
    return matches[0]
