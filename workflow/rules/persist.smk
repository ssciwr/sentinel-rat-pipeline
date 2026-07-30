"""Persist analysis results to PostgreSQL."""

rule persist:
    input:
        "data/results/.analyze_done"
    output:
        "data/results/.persist_done"
    shell:
        "echo 'persist rule scaffold' > {output}"
