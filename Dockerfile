FROM python:3.13-slim

SHELL ["/bin/bash", "-lc"]
ENV PYTHONUNBUFFERED=1

# uv installs the exact versions of uv.lock, which are also the ones tested in CI
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /bin/uv

WORKDIR /workflow

# git is needed to install sentinel-rat-dashboard (shared ORM models) from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# use the Python of the image, and copy packages instead of linking from the cache
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    # `python` is the one of the project environment, also for commands given by
    # docker compose (e.g. the daily-analysis service)
    PATH=/workflow/.venv/bin:$PATH \
    PYTHONPATH=/workflow/src

# dependencies first, so this layer is only rebuilt when uv.lock changes;
# --locked fails the build if uv.lock is out of date with pyproject.toml
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

COPY src/ ./src/
RUN uv sync --locked

EXPOSE 8001

CMD ["python", "-m", "sentinel_rat_pipeline.main"]
