FROM python:3.13-slim

SHELL ["/bin/bash", "-lc"]
ENV PYTHONUNBUFFERED=1

WORKDIR /workflow

ENV PYTHONPATH=/workflow/src

# git is needed to install sentinel-rat-dashboard (shared ORM models) from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY src/ ./src/
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

EXPOSE 8001

CMD ["python", "-m", "sentinel_rat_pipeline.main"]

