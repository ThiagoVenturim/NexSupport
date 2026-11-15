FROM python:3.12-slim

ARG EXTRAS="postgres,mysql"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DB_URL="sqlite:////data/support_engine.db"

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip && \
    pip install ".[${EXTRAS}]" && \
    useradd --create-home --shell /bin/bash support && \
    mkdir -p /data && \
    chown -R support:support /app /data

USER support
VOLUME ["/data"]

ENTRYPOINT ["support-engine"]
CMD ["--help"]
