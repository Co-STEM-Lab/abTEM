# Use ROCm build container on Ubuntu 24.04 as base
ARG ROCM_VERSION="7.2"
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete 
ENV ROCM_VERSION="${ROCM_VERSION}"
ENV INSTALL_DIR=/opt/software
ENV BUILD_THREADS=16

RUN echo "Starting build of dev container with ROCm ${ROCM_VERSION}"

# Ensure system packages are up to date and install global dependencies
RUN << 'EOF'
apt-get update
apt-get -y full-upgrade
apt-get install -y git \
    libfftw3-dev \
    libfftw3-mpi-dev \
    libopenblas-dev \
    libopenblas64-dev \
    openmpi-bin \
    libopenmpi-dev \
    libscalapack-openmpi-dev
                   
EOF


# Set up Python virtual environment and upgrade pip and setuptools
# Copy uv into image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

USER ubuntu
WORKDIR /home/ubuntu

# Ensure the installed binary is on the `PATH`
ENV PATH="/home/ubuntu/.local/bin/:$PATH"

ENV VIRTUAL_ENV="/home/ubuntu/venv"
RUN uv venv $VIRTUAL_ENV --python 3.12
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

RUN << 'EOF'
mv "${VIRTUAL_ENV}/bin/pip" "${VIRTUAL_ENV}/bin/pip-native"

cat > ${VIRTUAL_ENV}/bin/pip << 'PIP_WRAPPER'
#!/usr/bin/env bash
# Wrapper: redirect pip calls to uv pip for this venv

VENV_DIR="${VIRTUAL_ENV}"
exec uv pip --no-managed-python --no-cache "$@"
PIP_WRAPPER

chmod +x ${VIRTUAL_ENV}/bin/pip
EOF

RUN uv pip install ase notebook matplotlib fastrlock
RUN uv pip install amd-cupy --extra-index-url https://pypi.amd.com/rocm-7.2.0/simple
ENV ROCM_HOME="/opt/rocm"

USER root

RUN echo $PATH && ls -ahl ${VIRTUAL_ENV}/bin

# Get Libxc installed
ENV LIBXC_VERSION=7.0.0
COPY install-libxc-ubuntu.sh /tmp/
#RUN ls -la /tmp/install-libxc* && file /tmp/install-libxc-ubuntu.sh && head -5 /tmp/install-libxc-ubuntu.sh && bash -n /tmp/install-libxc-ubuntu.sh
#RUN set -x && /tmp/install-libxc-ubuntu.sh || { echo "Failed with exit $$?"; exit 1; }
RUN chmod +x /tmp/install-libxc-ubuntu.sh && /tmp/install-libxc-ubuntu.sh

ENV LIBXC_HOME=${INSTALL_DIR}/libxc
ENV LIBXC_INCLUDE=${INSTALL_DIR}/libxc/include
ENV LIBXC_LIBS=${INSTALL_DIR}/libxc/lib
ENV LD_LIBRARY_PATH=${LIBXC_LIBS}:${LD_LIBRARY_PATH:-}

# Install Libvdwxc
ENV LIBVDWXC_VERSION=0.5.0
COPY install-libvdwxc-ubuntu.sh /tmp/
RUN chmod +x /tmp/install-libvdwxc-ubuntu.sh && /tmp/install-libvdwxc-ubuntu.sh

ENV LIBVDWXC_HOME=${INSTALL_DIR}/libvdwxc
ENV LIBVDWXC_INCLUDE=${LIBVDWXC_HOME}/include
ENV LIBVDWXC_LIBS=${LIBVDWXC_HOME}/lib
ENV LD_LIBRARY_PATH=${LIBVDWXC_LIBS}:${LD_LIBRARY_PATH:-}


# Clean APT cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo 'ubuntu:ubuntu' | chpasswd
USER ubuntu

COPY siteconfig.py /home/ubuntu/
RUN ls -ahl ./* && GPAW_BUILD_GPU=0 GPAW_CONFIG='/home/ubuntu/siteconfig.py' uv pip install --no-cache-dir gpaw && rm siteconfig.py

# Install Claude AI tool
RUN curl -fsSL https://claude.ai/install.sh | bash

# This is a supposed fix to help with gfx1151
# See: https://github.com/pytorch/pytorch/issues/164346
ENV HSA_OVERRIDE_GFX_VERSION="11.0.0"

