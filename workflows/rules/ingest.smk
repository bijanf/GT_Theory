# Ingest rules: synthetic-profile generation for the smoke-synthetic target,
# and real-archive ingestion for the smoke-10 / smoke-100 / full subsets.

OUTPUTS = Path(config["outputs_root"])


# Wildcard constraints disambiguate the two ingest rules: synthetic only
# applies to the smoke-synthetic subset, real applies to anything else.
wildcard_constraints:
    real_subset=r"(?!smoke-synthetic\b)[A-Za-z0-9_\-]+",


rule make_synthetic_profile:
    """Forward-simulate a known GST history through column_1d and write the
    resulting present-day depth profile as a parquet file.  The recipe
    parameters are deterministic per ``site_id``, derived from the
    ``seed_base`` config so that re-runs are reproducible."""
    output:
        OUTPUTS / "smoke-synthetic" / "profiles" / "{site_id}.parquet",
    params:
        kappa=config["kappa_m2_s"],
        seed_base=config["seed_base"],
    log:
        OUTPUTS / "logs" / "make_synthetic_profile" / "{site_id}.log",
    shell:
        "python scripts/make_synthetic_profile.py "
        "--site-id {wildcards.site_id} "
        "--kappa {params.kappa} "
        "--seed-base {params.seed_base} "
        "--out {output} "
        "2>&1 | tee {log}"


rule ingest_real_profile:
    """Parse a Huang-Pollack archive file via the curated catalog and emit
    a uniform parquet for the downstream solve/invert rules.  Works for
    any non-synthetic subset (smoke-10, smoke-100, full, ...)."""
    output:
        OUTPUTS / "{real_subset}" / "profiles" / "{site_id}.parquet",
    params:
        catalog_path=workflow.source_path("../../catalogs/boreholes.yaml"),
    log:
        OUTPUTS / "logs" / "ingest_real_profile" / "{real_subset}__{site_id}.log",
    shell:
        "python scripts/ingest_real_profile.py "
        "--site-id {wildcards.site_id} "
        "--catalog {params.catalog_path} "
        "--out {output} "
        "2>&1 | tee {log}"
