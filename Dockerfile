# Use specific version of nvidia cuda image
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04



ENV DEBIAN_FRONTEND=noninteractive


# Set working directory
RUN mkdir /app
WORKDIR /app

# update ubuntu
RUN apt-get update -y && apt-get dist-upgrade -y

RUN apt-get install -y --no-install-recommends bash
# install binary dependencies:
# TODO: figure out which of these we actually need - this seems like massive overkill and is kind of a waste.
RUN apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libnvidia-common-565-server \
    libnvidia-compute-565-server \
    libnvidia-decode-565-server \
    libnvidia-decode-565-server \
    libnvidia-encode-565-server \
    libnvidia-encode-565-server \
    libnvidia-extra-565-server \
    libnvidia-gl-565-server \
    libx11-6 \
    nvidia-utils-565-server \
    software-properties-common \
    sudo \
    wget

# python3
RUN apt-get install -y --no-install-recommends \
    python3-full \
    python3-pip \
    python3-setuptools \
    python3-wheel





COPY requirements.txt /app/requirements.txt
RUN python3 -m venv /app/venv && \
    . /app/venv/bin/activate && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

COPY src/rp_handler.py /app/rp_handler.py
# we're using a virtual environment: we want to THAT python interpreter, not the system one.
# TODO: this seems like too many layers of indirection.
CMD [ "./venv/bin/python3.12", "rp_handler.py" ] 