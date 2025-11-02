# ---- Base Python image ----
FROM python:3.10-slim AS base

# Avoid python writing .pyc, ensure stdout flush
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- Install system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Set work directory ----
WORKDIR /app

# ---- Install Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy training code ----
COPY training/train_ci.py .

# ---- Set default entrypoint ----
ENTRYPOINT ["python", "train_ci.py"]