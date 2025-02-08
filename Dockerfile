# Use specific version of nvidia cuda image
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Set working directory
WORKDIR /

# Update and upgrade the system packages (Worker Template)
RUN apt update -y && \
    apt upgrade -y && \
    apt full-upgrade -y && \
    apt install -y --yes --no-install-recommends sudo ca-certificates git wget curl bash libgl1 libx11-6 software-properties-common ffmpeg build-essential \
    python3.12-dev python3.12-venv python3-pip \
    libnvidia-compute-565-server \
    libnvidia-encode-565-server \
    libnvidia-decode-565-server \
    libnvidia-extra-565-server \
    libnvidia-gl-565-server \
    libnvidia-encode-565-server \
    libnvidia-decode-565-server \
    libnvidia-common-565-server \
    nvidia-utils-565-server \
    && \
    apt autoremove -y && \
    apt clean -y && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment, then install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN python3.12 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Copy and run script to fetch models
COPY builder/fetch_models.py /fetch_models.py
RUN . /opt/venv/bin/activate && \
    python /fetch_models.py && \
    rm /fetch_models.py

# Copy source code into image
COPY src .

# Set default command
CMD . /opt/venv/bin/activate && python /rp_handler.py
