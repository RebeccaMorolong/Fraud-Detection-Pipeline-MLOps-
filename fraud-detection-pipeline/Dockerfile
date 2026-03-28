# ============================================================
# Stage 1: Builder — install dependencies with build tools
# ============================================================
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# --user installs to ~/.local so we can copy just that directory
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2: Runtime — lean final image (~600MB vs ~2GB)
# ============================================================
FROM python:3.11-slim AS runtime

# Copy only installed packages from builder (not the build tools)
COPY --from=builder /root/.local /root/.local

WORKDIR /app

# Copy application source
COPY src/ ./src/

# Create directories expected by the app
RUN mkdir -p models/saved monitoring data/raw data/processed

# Non-root user for security
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 8000

# Health check — Docker daemon monitors this
# --start-period=30s: don't check for 30s (startup time)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]