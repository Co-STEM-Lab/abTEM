#!/bin/bash

set -euxo pipefail

echo $PATH
which python
which pip

# libxc {version} module - DFT exchange-correlation functionals library

# Install build dependencies
# build-essential, gfortran         provides gfortran, gcc, g++, make
# autoconf automake libtool         provides full Autotools toolchain
apt-get install -y -q \
    wget \
    autoconf automake libtool \
    build-essential gfortran \
    cmake

pip install setuptools pytest

# Set installation prefix
LIBXC_PREFIX=${INSTALL_DIR}/libxc-${LIBXC_VERSION}

# Download and extract
cd /tmp
wget https://gitlab.com/libxc/libxc/-/archive/${LIBXC_VERSION}/libxc-${LIBXC_VERSION}.tar.gz
tar xf libxc-${LIBXC_VERSION}.tar.gz
cd libxc-${LIBXC_VERSION}

#export CFLAGS="{build_flags_c}"
#export FFLAGS="{build_flags_f}"

# Configure with optimizations
autoreconf -i
./configure --prefix=${LIBXC_PREFIX} \
    --enable-shared \
    --disable-static \
    --enable-fortran

# Build and install
make -j ${BUILD_THREADS}
make check
make install

# install python bindings
python setup.py install

# Create symlink for easy reference
ln -sf ${LIBXC_PREFIX} ${INSTALL_DIR}/libxc

# Cleanup
cd /tmp
rm -rf libxc-${LIBXC_VERSION}*

echo "libxc {version} installed successfully"

