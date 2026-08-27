import os
from pathlib import Path

mpi = True
if mpi:
    compiler = 'mpicc'

if '-fopenmp' not in extra_compile_args:
    extra_compile_args += ['-fopenmp']

if '-fopenmp' not in extra_link_args:
    extra_link_args += ['-fopenmp']

print(extra_compile_args)
print(extra_link_args)

build_flags = '{build_flags_c}'

for x in build_flags.strip(' ').split('-')[1:]:
    flag = '-' + x.strip()
    extra_compile_args += [flag]
    extra_link_args += [flag]

my_includes = [
    os.getenv('OPENBLAS_INCLUDE', ''),
    os.getenv('SCALAPACK_INCLUDE', ''),
    os.getenv('LIBXC_INCLUDE', ''),
    os.getenv('LIBVDWXC_INCLUDE', ''),
    os.getenv('FFTW_INCLUDE', ''),
]

my_libs = [
    os.getenv('OPENBLAS_LIBS', ''),
    os.getenv('SCALAPACK_LIBS', ''),
    os.getenv('LIBXC_LIBS', ''),
    os.getenv('LIBVDWXC_LIBS', ''),
    os.getenv('FFTW_LIBS', ''),
]

for minc in my_includes:
    if minc not in include_dirs and minc != '':
        include_dirs += [minc]

for mlib in my_libs:
    if mlib not in library_dirs and mlib != '':
        library_dirs += [mlib]

    if mlib not in runtime_library_dirs and mlib != '':
        runtime_library_dirs += [mlib]

###################
# SCALAPACK
###################
if 'scalapack' not in libraries or 'scalapack-openmpi' not in libraries:
    tmpdir = Path('/usr/lib/x86_64-linux-gnu')
    scalapack = True
    blacs = True
    if (tmpdir / 'libscalapack-openmpi.so').exists():
        libraries += ['scalapack-openmpi']
    elif (tmpdir / 'libscalapack.so').exists():
        libraries += ['scalapack']

if 'openblas' not in libraries:
    libraries += ['openblas']

if 1:
    libxc = True
    if 'xc' not in libraries:
        libraries += ['xc']


if 1:
    libvdwxc = True
    if 'vdwxc' not in libraries:
        libraries += ['vdwxc']


###################
# FFTW3
###################
if 1:
    tmpdir = Path('/lib/x86_64-linux-gnu')
    fftw = True

    # Prefer OMP threaded version
    if 'fftw3_omp' not in libraries:
        if (tmpdir / 'libfftw3_omp.so').exists():
            libraries += ['fftw3_omp']
    elif 'fftw3' not in libraries:
        if (tmpdir / 'libfftw3.so').exists():
            libraries += ['fftw3']

    if 'fftw3_mpi' not in libraries:
        if (tmpdir / 'libfftw3_mpi.so').exists():
            libraries += ['fftw3_mpi']


# hip
if os.getenv('GPAW_BUILD_GPU', '0') == '1':
    gpu = True
    gpu_target = 'hip-amd'
    gpu_compiler = 'hipcc'
    gpu_include_dirs = ['/opt/rocm/include']
    gpu_library_dirs = ['/opt/rocm/lib']
    gpu_compile_args = [
        '-g',
        '-O3',
        '--offload-arch=gfx1151',
       ]
    libraries += ['amdhip64', 'hipblas']


#if 'blacs' not in libraries:
#    libraries += ['blacs']


