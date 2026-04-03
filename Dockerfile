FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV TZ=UTC
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

COPY . .

RUN mkdir -p /app/logs /app/data

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python healthcheck.py || exit 1

ENTRYPOINT ["python", "-u", "main.py"]
