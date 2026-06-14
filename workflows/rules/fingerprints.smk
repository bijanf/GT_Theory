# Fingerprint rules.
#
# The F1-F5 implementations land in src/gt_theory/fingerprints/ in #23; for
# now, gather_per_site aggregates the per-site inversions into a single
# parquet summary so the DAG closes end-to-end.

OUTPUTS = Path(config["outputs_root"])


rule gather_per_site:
    """Concatenate per-site inversion summaries into one parquet."""
    input:
        lambda wc: expand(
            OUTPUTS / wc.subset / "inversions" / "{site_id}.parquet",
            site_id=site_ids_for(wc.subset),
        ),
    output:
        OUTPUTS / "{subset}" / "summary.parquet",
    log:
        OUTPUTS / "logs" / "gather_per_site" / "{subset}.log",
    shell:
        "python scripts/gather_per_site.py --inputs {input} --out {output} 2>&1 | tee {log}"
