"""Sentinel Rat Pipeline.

Snakemake workflow for watching filesystem, triggering ML analysis,
and persisting results to PostgreSQL.
"""

include: "workflow/rules/watch.smk"
include: "workflow/rules/analyze.smk"
include: "workflow/rules/persist.smk"
