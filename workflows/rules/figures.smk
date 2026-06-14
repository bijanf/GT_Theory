# Figure-rendering rules.
#
# Authored progressively in #21 (Fig 2 map), #23 (Figs 3-4-6 from F1-F5),
# #24 (Fig 5 regime diagrams), #25 (Fig 7 gap fingerprints), #26 (ED1-ED4),
# and #27 (Fig 1 schematic).  For now we emit a sentinel file so the DAG
# closes.

OUTPUTS = Path(config["outputs_root"])


rule figures_sentinel:
    input:
        OUTPUTS / config["subset"] / "summary.parquet",
    output:
        OUTPUTS / "figures" / ".sentinel",
    log:
        OUTPUTS / "logs" / "figures_sentinel.log",
    shell:
        "mkdir -p $(dirname {output}) "
        '&& echo "figures placeholder built from {input}" > {output} '
        "2>&1 | tee {log}"
