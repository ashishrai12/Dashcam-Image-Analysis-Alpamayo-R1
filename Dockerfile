# Use NVIDIA CUDA 12.4.1 base image (lighter and more standard for PyTorch 2.4)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV MODULAR_HOME="/root/.modular"
ENV PATH="/root/.modular/bin:/root/.modular/pkg/packages.modular.com_mojo/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Simple pip installation without complex flags
# Using Ubuntu 22.04 avoids the PEP 668 'break-system-packages' issue entirely
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# Copy and install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Modular CLI
RUN curl -fsSL https://get.modular.com | sh -

# Install Mojo
RUN --mount=type=secret,id=MODULAR_AUTH_TOKEN \
    if [ -f /run/secrets/MODULAR_AUTH_TOKEN ]; then \
    MODULAR_AUTH_TOKEN=$(cat /run/secrets/MODULAR_AUTH_TOKEN) && \
    modular auth $MODULAR_AUTH_TOKEN && \
    modular install mojo; \
    else \
    echo "Warning: MODULAR_AUTH_TOKEN not found. Skipping Mojo installation."; \
    fi

# Copy code
COPY . .

# Default command
CMD ["mojo", "src/main.mojo"]
