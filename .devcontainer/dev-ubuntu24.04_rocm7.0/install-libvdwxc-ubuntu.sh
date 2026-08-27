#!/bin/bash

set -euxo pipefail

# libxc {version} module - DFT exchange-correlation functionals library

apt-get install -y -q \
    wget \
    autoconf automake libtool \
    build-essential gfortran \
    libfftw3-dev libfftw3-mpi-dev \
    openmpi-bin libopenmpi-dev
    

apt-get clean && rm -rf /var/lib/apt/lists/*

# Set installation prefix
LIBVDWXC_PREFIX=${INSTALL_DIR}/libvdwxc-${LIBVDWXC_VERSION}

# Clone source
cd /tmp
rm -rf libvdwxc
wget https://gitlab.com/libvdwxc/libvdwxc/-/archive/${LIBVDWXC_VERSION}/libvdwxc-${LIBVDWXC_VERSION}.tar.gz
tar xzf libvdwxc-${LIBVDWXC_VERSION}.tar.gz
cd libvdwxc-${LIBVDWXC_VERSION}

# Build libvdwxc
sh autogen.sh
autoreconf -i
mkdir build && cd build

unset CC
unset FC

export CFLAGS="-O3 -march=native -mtune=native"
export FCFLAGS="-g -O2 -march=native -mtune=native"
#BUILD_FFTW3_INCLUDES="/include"
#BUILD_FFTW3_LIBS="/lib/x86_64-linux-gnu"

../configure --prefix=${LIBVDWXC_PREFIX} \
    CC="mpicc" FC="mpif90" \
    --with-fftw3 
#    FFTW3_INCLUDES="-I${BUILD_FFTW3_INCLUDES}" \
#    FFTW3_LIBS="-L${BUILD_FFTW3_LIBS} -lfftw3_omp -lfftw3_mpi"

make -j ${BUILD_THREADS}
make check
make install

ln -sf ${LIBVDWXC_PREFIX} ${INSTALL_DIR}/libvdwxc
# Cleanup
cd /tmp
rm -rf libvdwxc-${LIBVDWXC_VERSION}*

echo "✓ libvdwxc ${LIBVDWXC_VERSION} installed to ${LIBVDWXC_PREFIX}"

