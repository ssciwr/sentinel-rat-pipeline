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

Given `watched-images/CAM01_image_20260728T170840Z.jpg`:

### 1. Watch
`watchdog` detects the new image. The first new image opens a 1-minute window (configurable via `BATCH_WINDOW_SECONDS`); all images that arrive within that window are collected and processed together as one batch. Batches larger than `MAX_BATCH_SIZE` (default 50) are split into several requests.

Images already in the watch folder at startup are processed as well (in batches of at most `MAX_BATCH_SIZE`); set `PROCESS_BACKLOG=false` to skip them. The watcher starts before the folder is listed, so images added meanwhile are not missed; images already stored in the database are skipped, and images still being copied are processed once the copy has finished. Each image is processed only once, even if it is seen by both the backlog and the watcher.

### 2. Analyze
The pipeline calls `sentinel-rat-ml-pipeline` at `/api/v1/analyze` once per batch, with a list of image paths:

```json
{"image_paths": ["CAM01_image_20260728T170840Z.jpg", "CAM01_image_20260728T170905Z.jpg"]}
```

The service returns one result per image, each carrying its `image_path`:

```json
{"results": [{"image_path": "CAM01_image_20260728T170840Z.jpg", "detections": [...]}, ...]}
```

Each result is then persisted on its own, so a failing image does not drop the rest of the batch.

### 3. Persist
Each image result is written in its own transaction into the tables of the [sentinel-rat-dashboard](https://github.com/ssciwr/sentinel-rat-dashboard) data model. For example, the result

```json
{
  "image_path": "CAM01_image_20260728T170840Z.jpg",
  "detection_model": {"name": "megadetector", "version": "v5a"},
  "classification_model": {"name": "rodent-classifier", "version": "0.1"},
  "detections": [
    {
      "detected_class": "animal",
      "confidence": 0.92,
      "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.5},
      "classifications": [
        {"genus": "Rattus", "species": "rattus", "common_name": "black rat", "confidence": 0.87},
        {"genus": "Rattus", "species": "norvegicus", "common_name": "brown rat", "confidence": 0.1}
      ]
    }
  ]
}
```

is stored as:

`image_capture` — one row per image. The camera is looked up by the file name prefix (`CAM01`) and must already exist in the `camera` table; `captured_at` comes from the file name, else from EXIF, else the current time; `location` is the camera's current location.

| id | camera_id | image_path | captured_at | uploaded_at | location |
|----|-----------|------------|-------------|-------------|----------|
| 1 | 1 | CAM01_image_20260728T170840Z.jpg | 2026-07-28 17:08:40+00 | 2026-10-07 09:15:02+00 | POINT(80.6 7.3) |

`object_detection` — one row per detected object.

| id | image_capture_id | det_model_id | confidence | bbox | detected_class |
|----|------------------|--------------|------------|------|----------------|
| 1 | 1 | 1 | 0.92 | {"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.5} | animal |

`species_classification` — one row per species candidate of a detection.

| id | object_detection_id | taxonomy_id | clas_model_id | confidence |
|----|---------------------|-------------|---------------|------------|
| 1 | 1 | 1 | 2 | 0.87 |
| 2 | 1 | 2 | 2 | 0.1 |

`ml_model` and `taxonomy` — looked up by name/version/task and genus/species, and created only if missing, so they are shared across images.

| id | name | version | task |
|----|------|---------|------|
| 1 | megadetector | v5a | detection |
| 2 | rodent-classifier | 0.1 | classification |

| id | genus | species | common_name |
|----|-------|---------|-------------|
| 1 | Rattus | rattus | black rat |
| 2 | Rattus | norvegicus | brown rat |

When an image is deleted from the watch folder, its `image_capture` row is deleted together with its detections, classifications and corrections (*to be re-considered later to keep a history of deleted images*).
