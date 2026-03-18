# --- Build Stage ---
FROM python:3.10-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Final Runtime Stage ---
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000
ENV TZ=Asia/Manila

WORKDIR /app

# Install runtime system dependencies
# libgl1 and libglib2.0-0 are for OpenCV/MediaPipe
# curl is for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libwebkit2gtk-4.1-0 \
    libayatana-appindicator3-1 \
    librsvg2-common \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create necessary directories for persistence
RUN mkdir -p data/offline data/faces static/uploads

# Expose the application port
EXPOSE 10000

# Health check to ensure the service is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/ || exit 1

# Start the application using Uvicorn
# Note: We use 0.0.0.0 to allow external access within the Docker network
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
