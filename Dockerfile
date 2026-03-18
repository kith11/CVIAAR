# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies in a single layer to keep it compact
# libgl1 and libglib2.0-0 are required for MediaPipe and OpenCV
# libgtk-3-0 and others are for pywebview (if running kiosk in GUI mode)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libwebkit2gtk-4.1-0 \
    libayatana-appindicator3-1 \
    librsvg2-common \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first for better performance
RUN pip install --upgrade pip

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install Python packages using a build cache mount to speed up subsequent builds
# We use --default-timeout to prevent connection issues from stalling the build
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose port 10000 (standard for Render) or the port provided by environment
EXPOSE 10000

# Run the application using Uvicorn. 
# We use a shell to allow environment variable expansion for $PORT.
# Render automatically sets $PORT to 10000 or a dynamic value.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
