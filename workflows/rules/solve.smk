# Forward solver + Bayesian inversion, per site.  These are the rules
# that get fanned out to PIK HPC by the pik-hpc Snakemake profile; on the
# laptop they run locally.

OUTPUTS = Path(config["outputs_root"])


rule invert_profile:
    """Bootstrap-Tikhonov inversion of one borehole profile -> GST history
    posterior summary as parquet."""
    input:
        profile=OUTPUTS / "{subset}" / "profiles" / "{site_id}.parquet",
    output:
        OUTPUTS / "{subset}" / "inversions" / "{site_id}.parquet",
    wildcard_constraints:
        subset=r"[A-Za-z0-9_\-]+",
    params:
        n_bootstrap=config["n_bootstrap"],
        seed_base=config["seed_base"],
    resources:
        time_min=120,
        mem_mb=4000,
        slurm_partition="standard",
    log:
        OUTPUTS / "logs" / "invert_profile" / "{subset}__{site_id}.log",
    shell:
        "python scripts/invert_profile.py "
        "--site-id {wildcards.site_id} "
        "--profile {input.profile} "
        "--n-bootstrap {params.n_bootstrap} "
        "--seed-base {params.seed_base} "
        "--out {output} "
        "2>&1 | tee {log}"
