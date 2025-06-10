# Use NVIDIA PyTorch base image with CUDA support
ARG CUDA_VERSION=12.6
FROM pytorch/pytorch:2.7.0-cuda${CUDA_VERSION}-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# Install system dependencies including bash
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    sudo \
    bash \
    bash-completion \
    && rm -rf /var/lib/apt/lists/*

# Install only the essential libraries with pip (simpler and more reliable)
RUN pip install --no-cache-dir \
    torch torchvision pytorch-lightning \
    matplotlib Pillow pyyaml pandas seaborn \
    tqdm scikit-learn tensorboard scipy

# Create a non-root user with same UID/GID as host user (typically 1000)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -G sudo -m -s /bin/bash appuser
RUN echo 'appuser ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Set working directory
WORKDIR /app

# Create necessary directories and set ownership
RUN mkdir -p /app/logs /app/checkpoints /app/predictions /app/cache/torch \
    /home/appuser/.vscode-server /home/appuser/.vscode-server-insiders && \
    chown -R appuser:appuser /home/appuser /app

# Create an entrypoint script to fix permissions at runtime
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'sudo chown -R appuser:appuser /app' >> /entrypoint.sh && \
    echo 'sudo chmod -R u+w /app' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Switch to non-root user
USER appuser

# Set bash as default shell
SHELL ["/bin/bash", "-c"]

# Use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash", "-l"]