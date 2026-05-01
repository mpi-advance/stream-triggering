#!/bin/bash

CYAN='\033[0;36m'
RESET='\033[0m'

usage() {
    echo "Usage: $0 [-T value] [-B value]"
    echo " -T Integer that specifies build system: 0 (Tioga, default), 1 (Tuoloumne), 2 (Frontier)"
    echo " -B Build mode: 0 (Debug, default), 1 (Release), 2 (RelWithDebInfo)"
    echo " -R String for full rocm module name (default is \"rocm\")"
    echo " -L String for libfabric version number (Possible values: 1.22.0, 2.1, SYSTEM)"
}

while getopts ":T:B:R:L:" opt; do
    case $opt in
        T)
            VERSION="$OPTARG"
            ;;
        B)
            BUILD_MODE="$OPTARG"
            ;;
        R)
            ROCM_VERSION="$OPTARG"
            ;;
        L)
            LIBFABRIC_VERSION="$OPTARG"
            ;;
        *)
            usage
            exit
            ;;
    esac
done

set -e

if [ -z $VERSION ]; then
    VERSION=0
fi

# Prepare System Specific variables
if [ "$VERSION" -eq 0 ]; then
    SYSTEM=tioga
    GPU_ARCH=gfx90a
    LIBFABRIC_DEFAULT=2.1
elif [ "$VERSION" -eq 1 ]; then
    SYSTEM=tuolumne
    GPU_ARCH=gfx942
    LIBFABRIC_DEFAULT=2.1
elif [ "$VERSION" -eq 2 ]; then
    SYSTEM=frontier
    GPU_ARCH=gfx90a
    LIBFABRIC_DEFAULT=1.22.0
    EXTRA_MODULES=cce/20.0.0
else
    echo "Invalid system specified, stopping."
    exit 1
fi

echo -e "Running ${CYAN}$SYSTEM${RESET} version"

# P
if [ -z $ROCM_VERSION ]; then
    ROCM_VERSION="rocm"
fi 

if [ -z $LIBFABRIC_VERSION ]; then
    LIBFABRIC_MODULE="libfabric/${LIBFABRIC_DEFAULT}"
    LIBFABRIC_DIR="/opt/cray/libfabric/${LIBFABRIC_DEFAULT}"
elif [ "$LIBFABRIC_VERSION" = "SYSTEM" ]; then
    LIBFABRIC_DIR="/usr/lib64"
else
    LIBFABRIC_MODULE="libfabric/${LIBFABRIC_VERSION}"
    LIBFABRIC_DIR="/opt/cray/libfabric/${LIBFABRIC_VERSION}"
fi

# Load Modules
module load ${EXTRA_MODULES} "craype-accel-amd-${GPU_ARCH}" "${ROCM_VERSION}" ${LIBFABRIC_MODULE}
module list

# Extra setup for Frontier
if [ "$VERSION" -eq 2 ]; then
    export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
    CMAKE_EXTRAS=( "-DCMAKE_EXE_LINKER_FLAGS=${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}" )
fi

if [ -z $BUILD_MODE ]; then
    BUILD_MODE=0
fi

if [ "$BUILD_MODE" -eq 0 ]; then
    MODE="Debug"
elif [ "$BUILD_MODE" -eq 1 ]; then
    MODE="Release"
elif [ "$BUILD_MODE" -eq 2 ]; then
    MODE="RelWithDebInfo"
fi

echo -e "Building in ${CYAN}${MODE}${RESET} mode"

DIR_TO_BUILD="build"
if [ -d $DIR_TO_BUILD ]; then
	rm -rf $DIR_TO_BUILD
fi
mkdir $DIR_TO_BUILD && cd $DIR_TO_BUILD

cmake -DUSE_HIP_BACKEND=ON -DUSE_CXI_BACKEND=ON -DLIBFABRIC_PREFIX=${LIBFABRIC_DIR} \
      -DCMAKE_HIP_ARCHITECTURES=${GPU_ARCH} -DCMAKE_INSTALL_PREFIX=${HOME}/apps/stream_trigger \
      "${CMAKE_EXTRAS[@]}" -DCMAKE_BUILD_TYPE=$MODE ..

make -j8
make install

