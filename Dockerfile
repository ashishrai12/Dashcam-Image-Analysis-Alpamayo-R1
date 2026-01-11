# Use a lightweight CUDA base image since Alpamayo-R1 is a heavy VLA model
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and uv (the fastest way to handle this repo's dependencies)
RUN apt-get update && apt-get install -y python3 python3-pip curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv

# Copy only dependency files first (for better caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy the rest of the application
COPY . .

# Default command (adjust 'main.py' to the actual entry script, e.g., test_inference.py)
CMD ["uv", "run", "python", "test_inference.py"]
