#!/bin/bash

set -e

DIR_TO_BUILD="build"
if [ -d $DIR_TO_BUILD ]; then
	rm -rf $DIR_TO_BUILD
fi
mkdir $DIR_TO_BUILD && cd $DIR_TO_BUILD

cmake -DCMAKE_INSTALL_PREFIX=/g/g16/derek/apps/stream_trigger ..
make -j8
