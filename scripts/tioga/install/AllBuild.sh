#!/bin/bash

set -e

module load craype-accel-amd-gfx90a
module load rocm/6.3.1

if [ $# -eq 0 ]; then
    echo "Building in debug mode"
    MODE="Debug"
else
    echo "Building in release mode"
    MODE="Release"
fi

DIR_TO_BUILD="build"
if [ -d $DIR_TO_BUILD ]; then
	rm -rf $DIR_TO_BUILD
fi
mkdir $DIR_TO_BUILD && cd $DIR_TO_BUILD

cmake -DUSE_HIP_BACKEND=ON -DUSE_CXI_BACKEND=ON -DLIBFABRIC_PREFIX=/opt/cray/libfabric/2.1/ \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_INSTALL_PREFIX=/g/g16/derek/apps/stream_trigger \
      -DCMAKE_BUILD_TYPE=$MODE ..

make -j8
make install

