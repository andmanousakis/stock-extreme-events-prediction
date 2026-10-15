# Base image with Python 3.11.
FROM python:3.11.14-slim

# Link the image to its GitHub repo when pushed to GitHub Container Registry.
LABEL org.opencontainers.image.source=https://github.com/andmanousakis/stock-extreme-events-prediction

# Don't write __pycache__ folders into the mounted project.
ENV PYTHONDONTWRITEBYTECODE=1

# Print output immediately so container logs are live.
ENV PYTHONUNBUFFERED=1

# Let Python import src and utils from the project root.
ENV PYTHONPATH=/app

# Skip pip's download cache to keep the image smaller.
ENV PIP_NO_CACHE_DIR=1

# Hide pip's "new version available" message.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Give the non-root user a writable home (yfinance cache).
ENV HOME=/tmp

# Give matplotlib a writable config folder.
ENV MPLCONFIGDIR=/tmp/matplotlib

# Silence MLflow's warnings about git not being installed.
ENV GIT_PYTHON_REFRESH=quiet

# Work from the project root.
WORKDIR /app

# Copy only what the install needs, so code changes don't reinstall dependencies.
COPY pyproject.toml ./
COPY utils ./utils

# Install pinned dependencies and dev tools, using CPU-only torch.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"

# Copy the rest of the project.
COPY . .

# Run data retrieval by default.
CMD ["python", "-m", "src.1_data_retrieval"]
