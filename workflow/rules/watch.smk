"""Scan the watch folder for new images."""

rule watch:
    output:
        "data/results/.watch_done"
    shell:
        "echo 'watch rule scaffold' > {output}"
