# NEXUS Runtime — Multi-stage build for marine robotics edge deployment
#
# Usage:
#   docker build -t nexus-runtime .
#   docker run --rm nexus-runtime python -m jetson
#
# For Jetson (NVIDIA GPU):
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-runtime-ubuntu22.04 -t nexus-runtime:jetson .

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS runtime

LABEL maintainer="NEXUS Team"
LABEL description="NEXUS Distributed Intelligence Platform for Marine Robotics"

WORKDIR /opt/nexus

# System dependencies for serial, MQTT, and sensor I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make cmake \
    libserial-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY pyproject.toml .
COPY nexus/ nexus/
COPY jetson/ jetson/
COPY hardware/ hardware/
COPY tests/ tests/

# Install Python dependencies
RUN pip install --no-cache-dir numpy pyserial pytest ruff mypy 2>/dev/null || true

# Create non-root user for safety
RUN useradd -m -s /bin/bash nexus
USER nexus

ENV PYTHONPATH=/opt/nexus
ENV NEXUS_LOG_LEVEL=INFO

EXPOSE 8080 1883 5672

CMD ["python", "-m", "jetson"]
