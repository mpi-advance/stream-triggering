#!/bin/bash

compile_test()(
    cd ./$TEST_DIR
    set -x
    CC -D__HIP_PLATFORM_AMD__ -O3 -g -std=c++20 -x hip $3 ../$1 -I$ST_INC_PATH -L$ST_LIB_PATH -o $2 -lstream-triggering
)


set -e

module load craype-accel-amd-gfx90a
module load rocm/6.3.1

ST_PATH=/g/g16/derek/apps/stream_trigger
ST_LIB_PATH=$ST_PATH/lib
ST_INC_PATH=$ST_PATH/include

TEST_DIR=testing_dir
mkdir -p $TEST_DIR

PROGRAM=./pingpong.cpp
OUTPUT=cxi-coarse
DEFINES="-DCXI_BACKEND"
compile_test $PROGRAM $OUTPUT $DEFINES

OUTPUT=cxi-fine
DEFINES="-DCXI_BACKEND -DFINE_GRAINED_TEST"
compile_test $PROGRAM $OUTPUT "$DEFINES"

OUTPUT=hip-test
DEFINES="-DHIP_BACKEND"
compile_test $PROGRAM $OUTPUT $DEFINES

OUTPUT=thread-test
DEFINES="-DTHREAD_BACKEND"
compile_test $PROGRAM $OUTPUT $DEFINES

