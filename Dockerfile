FROM python:3.13-slim

SHELL ["/bin/bash", "-lc"]
ENV PYTHONUNBUFFERED=1

WORKDIR /workflow

ENV PYTHONPATH=/workflow/src

COPY workflow/ ./workflow/
COPY src/ ./src/
COPY Snakefile .
COPY pyproject.toml .

RUN pip install --no-cache-dir snakemake

RUN pip install --no-cache-dir -e .

RUN mkdir -p /workflow/data/results /workflow/data/logs

EXPOSE 8001

CMD ["python", "-m", "sentinel_rat_pipeline.main"]

