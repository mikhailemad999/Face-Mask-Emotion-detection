# Django Backend Dockerfile
# CUDA 12.1 + cuDNN 8 (for RTX 2060)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="FaceGuard.AI"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl gnupg2 && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 && \
    apt-get install -y python3 python3-dev unixodbc-dev libgl1-mesa-glx

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python3 manage.py collectstatic --no-input --settings=config.settings || true

EXPOSE 8000
