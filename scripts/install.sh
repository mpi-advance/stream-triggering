#!/bin/bash

CYAN='\033[0;36m'
RESET='\033[0m'

usage() {
    echo "Usage: $0 [-T value] [-B value]"
    echo " -T Integer that specifies build system: 0 (Tioga, default), 1 (Tuoloumne), 2 (Frontier)"
    echo " -B Build mode: 0 (Debug, default), 1 (Release), 2 (RelWithDebInfo)"
}

while getopts ":T:B:" opt; do
    case $opt in
        T)
            VERSION="$OPTARG"
            ;;
        B)
            BUILD_MODE="$OPTARG"
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

if [ "$VERSION" -eq 0 ]; then
    SYSTEM=tioga
    GPU_ARCH=gfx90a
    LIBFABRIC=/opt/cray/libfabric/2.1/
elif [ "$VERSION" -eq 1 ]; then
    SYSTEM=tuolumne
    GPU_ARCH=gfx942
    LIBFABRIC=/opt/cray/libfabric/2.1/
elif [ "$VERSION" -eq 2 ]; then
    SYSTEM=frontier
    GPU_ARCH=gfx90a
    LIBFABRIC=/opt/cray/libfabric/1.22.0/
else
    echo "Invalid system specified, stopping."
    exit 1
fi

echo -e "Running ${CYAN}$SYSTEM${RESET} version"
module load rocm "craype-accel-amd-${GPU_ARCH}"
module list

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

echo -e "Building in ${MODE} mode"

DIR_TO_BUILD="build"
if [ -d $DIR_TO_BUILD ]; then
	rm -rf $DIR_TO_BUILD
fi
mkdir $DIR_TO_BUILD && cd $DIR_TO_BUILD

cmake -DUSE_HIP_BACKEND=ON -DUSE_CXI_BACKEND=ON -DLIBFABRIC_PREFIX=/opt/cray/libfabric/2.1/ \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_INSTALL_PREFIX=${HOME}/apps/stream_trigger \
      -DCMAKE_BUILD_TYPE=$MODE ..

make -j8
make install

