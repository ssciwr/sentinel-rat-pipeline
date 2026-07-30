"""Call the ML pipeline service for analysis."""

rule analyze:
    input:
        "data/results/.watch_done"
    output:
        "data/results/.analyze_done"
    shell:
        "echo 'analyze rule scaffold' > {output}"
