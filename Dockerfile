FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential curl wget git unzip \
    ca-certificates software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /home

ADD . /home/TPU-FI
WORKDIR /home/TPU-FI
