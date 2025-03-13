#!/bin/bash

set -e

module load rocm

DIR_TO_BUILD="build"
if [ -d $DIR_TO_BUILD ]; then
	rm -rf $DIR_TO_BUILD
fi
mkdir $DIR_TO_BUILD && cd $DIR_TO_BUILD

cmake -DUSE_IMPLEMENTATION=CXI -DLIBFABRIC_PREFIX=/opt/cray/libfabric/2.1/ -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_C_COMPILER=hipcc -DCMAKE_INSTALL_PREFIX=/g/g16/derek/apps/stream_trigger ..
make -j8
