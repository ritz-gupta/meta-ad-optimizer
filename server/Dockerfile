FROM python:3.11-slim

WORKDIR /app

# curl needed for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer first — changes to source code won't invalidate this cache
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full package and install it
COPY . .
RUN pip install --no-cache-dir -e .

# HF Spaces runs containers as uid 1000 (non-root requirement)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER 1000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "meta_ad_optimizer.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
