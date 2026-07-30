# Sentinel Rat Pipeline

Work in progress...

## Overview

Snakemake workflow that watches a folder for new images, triggers `sentinel-rat-ml-pipeline` for analysis, and persists results to the PostgreSQL (PostGIS) database.

## Run with Docker Compose (recommended)

From `sentinel-rat/`:

```bash
docker compose up --build
```

The pipeline will watch `watched-images/` on the host and process new images automatically.

No local installation is needed. The container is built from `condaforge/mambaforge` with a dedicated conda environment that installs Snakemake via `conda install -c bioconda snakemake` and the package dependencies via `pip install -e .`.

## Run locally (optional)

```bash
conda create -n sentinel-rat-pipe python=3.13 -y
conda activate sentinel-rat-pipe
pip install -e .
snakemake --use-conda -s Snakefile watch analyze persist --cores 1
```

## Example workflow

Given `watched-images/a-cat-and-a-mouse-are-eating-food-together-photo.jpg`:

### 1. Watch
`watchdog` detects the new image and passes its path to the next rule.

### 2. Analyze
The pipeline calls `sentinel-rat-ml-pipeline` at `/api/v1/analyze`.

Response:

```json
{
  "image_path": "/data/images/a-cat-and-a-mouse-are-eating-food-together-photo.jpg",
  "animals_detected": 2,
  "species": {"cat": 1, "rodent": 1},
  "confidence": 0.95
}
```

### 3. Persist
The result is written to PostgreSQL as:

| id | detection_id | image_path | animals_detected | species | confidence | detected_at |
|----|--------------|------------|------------------|---------|------------|-------------|
| 1 | 20260730115638 | /data/images/...photo.jpg | 2 | {"cat": 1, "rodent": 1} | 0.95 | 1785412598.89|