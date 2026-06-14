"""Borehole-catalog loader.

The YAML catalog at ``catalogs/boreholes.yaml`` defines:

* a default archive root (overridable via the ``GT_THEORY_BOREHOLE_ROOT``
  environment variable);
* named subsets (e.g. ``smoke-10``) — ordered lists of site IDs;
* per-site curated metadata (lat, lon, max depth, QC tier, notes), which
  overrides values parsed from the raw archive headers.

This module exposes :func:`load_catalog`, :func:`subset_ids`, and
:func:`resolve_data_root` to keep callers decoupled from YAML internals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "catalogs" / "boreholes.yaml"
DATA_ROOT_ENV = "GT_THEORY_BOREHOLE_ROOT"


@dataclass(frozen=True)
class SiteMetadata:
    """Per-site catalog entry.  Mirrors a row in ``boreholes.yaml`` under
    ``sites:``."""

    site_id: str
    country: str
    lat_deg: float
    lon_deg: float
    max_depth_m: float
    measurement_year: float | None
    qc_tier: int
    notes: str


@dataclass(frozen=True)
class Catalog:
    schema_version: int
    default_data_root: str
    subsets: dict[str, list[str]] = field(default_factory=dict)
    sites: dict[str, SiteMetadata] = field(default_factory=dict)

    def subset_ids(self, name: str) -> list[str]:
        if name not in self.subsets:
            available = ", ".join(sorted(self.subsets)) or "<none>"
            raise KeyError(f"Subset {name!r} not in catalog (available: {available})")
        return list(self.subsets[name])

    def site(self, site_id: str) -> SiteMetadata:
        if site_id not in self.sites:
            raise KeyError(f"Site {site_id!r} not in catalog")
        return self.sites[site_id]


def _expand(text: str) -> str:
    """Expand both ``$VAR``/`${VAR}`` and ``~`` in a string from a YAML
    value, preserving the literal if no expansion applies."""
    return os.path.expandvars(os.path.expanduser(text))


def load_catalog(path: str | Path | None = None) -> Catalog:
    """Parse the borehole catalog YAML at *path* (default:
    ``catalogs/boreholes.yaml`` at the repo root) into a :class:`Catalog`.
    """
    p = Path(path or DEFAULT_CATALOG_PATH).expanduser().resolve()
    raw: dict[str, Any] = yaml.safe_load(p.read_text())

    schema_version = int(raw.get("schema_version", 1))
    data_root = str(raw.get("default_data_root", "data/raw/boreholes/huang2000"))
    subsets_raw = raw.get("subsets") or {}
    sites_raw = raw.get("sites") or {}

    subsets: dict[str, list[str]] = {}
    for name, ids in subsets_raw.items():
        if not isinstance(ids, list):
            raise ValueError(f"Subset {name!r} must be a list of site IDs.")
        subsets[name] = [str(x) for x in ids]

    sites: dict[str, SiteMetadata] = {}
    for site_id, meta in sites_raw.items():
        if not isinstance(meta, dict):
            raise ValueError(f"Site {site_id!r} entry must be a mapping.")
        sites[site_id] = SiteMetadata(
            site_id=site_id,
            country=str(meta.get("country", "")),
            lat_deg=float(meta["lat_deg"]),
            lon_deg=float(meta["lon_deg"]),
            max_depth_m=float(meta.get("max_depth_m", 0.0)),
            measurement_year=(
                float(meta["measurement_year"])
                if meta.get("measurement_year") is not None
                else None
            ),
            qc_tier=int(meta.get("qc_tier", 1)),
            notes=str(meta.get("notes", "")),
        )

    return Catalog(
        schema_version=schema_version,
        default_data_root=data_root,
        subsets=subsets,
        sites=sites,
    )


def resolve_data_root(catalog: Catalog) -> Path:
    """Return the resolved archive root, honouring the
    ``GT_THEORY_BOREHOLE_ROOT`` environment variable when set."""
    env = os.environ.get(DATA_ROOT_ENV)
    raw = env if env else _expand(catalog.default_data_root)
    return Path(_expand(raw)).expanduser().resolve()
