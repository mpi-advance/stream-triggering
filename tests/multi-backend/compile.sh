#!/bin/bash

ARROW='\033[0;34m--> \033[0m'

compile_test()(
    cd ./$TEST_DIR
    set -x
    CC -D__HIP_PLATFORM_AMD__ -O3 -g -std=c++20 -x hip $3 $1 -I$ST_INC_PATH -L$ST_LIB_PATH -o $2 -lstream-triggering
)

set -e

### Command-line options
usage() {
    echo "Usage: $0 [-dgXCHT] [-f file]"
    echo " -d Run with MI300A modules (default MI250X modules)"
    echo " -f [file] Compile file provided."
    echo " -g Add defines for cuda GPUs."
    echo " -X No CXI build"
    echo " -H No HIP build"
    echo " -C No CUDA build"
    echo " -T No THREAD build" 
}

# Parse command line options
DEVICE_DEFINE="-DNEED_HIP"
while getopts ":dgXCHTf:" opt; do
    case $opt in
        d)
            MI300A=1
            ;;
        g)
            DEVICE_DEFINE="-DNEED_CUDA"
            ;;
        X)
            SKIP_CXI=1
            ;;
        H)
            SKIP_HIP=1
            ;;
        C)
            SKIP_CUDA=1
            ;;
        T)
            SKIP_THREAD=1
            ;;
        f)
            PROGRAM="$OPTARG"
            ;;
        *)
            usage
            exit
            ;;
    esac
done

### Check and Prepare filename output

if [ -z $PROGRAM ]; then
    echo "No file provided. Stopping."
    exit 1
fi

if [ ! -e $PROGRAM ]; then
    echo "Could not find: $PROGRAM. Stopping."
    exit 1
fi

BASE_OUTPUT=${PROGRAM%.*}
BASE_OUTPUT=${BASE_OUTPUT##*/}

### Modules and device settings
DEVICE="_"
if [ ! -v MI300A ]; then
    echo " -> Running for the MI250X"
    module load craype-accel-amd-gfx90a
    DEVICE+="TIOGA"
else
    echo " -> Running for the MI300A"
    module load craype-accel-amd-gfx942
    DEVICE+="TUOLUMNE"
fi

module load rocm

### Prepare output directories

SCRATCH=../scratch
EXECS=$SCRATCH/execs
FLUX=$SCRATCH/flux
OUTPUT=$SCRATCH/output
TMP_DIR=$SCRATCH/tmp

if [ ! -d $SCRATCH ]; then
    echo " -> Making scratch directory"
    mkdir $SCRATCH
fi

if [ ! -d $EXECS ]; then
    echo " -> Making executables directory"
    mkdir $EXECS
fi

if [ ! -d $FLUX ]; then
    echo " -> Making flux directory"
    mkdir $FLUX
fi

if [ ! -d $OUTPUT ]; then
    echo " -> Making outputs directory"
    mkdir $OUTPUT
fi

if [ ! -d $TMP_DIR ]; then
    echo " -> Making temp directory"
    mkdir $TMP_DIR
fi

###

BASE_OUTPUT+=$DEVICE
BASE_OUTPUT=$EXECS/$BASE_OUTPUT
echo " -> $PROGRAM -> $BASE_OUTPUT"

ST_PATH=/g/g16/derek/apps/stream_trigger
ST_LIB_PATH=$ST_PATH/lib
ST_INC_PATH=$ST_PATH/include

if [ ! -v SKIP_CXI ]; then
    OUTPUT=$BASE_OUTPUT"_cxi-coarse"
    echo -e "$ARROW$OUTPUT"
    DEFINES="-DCXI_BACKEND $DEVICE_DEFINE"
    compile_test $PROGRAM $OUTPUT "$DEFINES"

    OUTPUT=$BASE_OUTPUT"_cxi-fine"
    echo -e "$ARROW$OUTPUT"
    DEFINES="-DCXI_BACKEND -DFINE_GRAINED_TEST $DEVICE_DEFINE"
    compile_test $PROGRAM $OUTPUT "$DEFINES"
fi

if [ ! -v SKIP_HIP ]; then
    OUTPUT=$BASE_OUTPUT"_hip"
    echo -e "$ARROW$OUTPUT"
    DEFINES="-DHIP_BACKEND $DEVICE_DEFINE"
    compile_test $PROGRAM $OUTPUT "$DEFINES"
fi

if [ ! -v SKIP_CUDA ]; then
    OUTPUT=$BASE_OUTPUT"_cuda"
    echo -e "$ARROW$OUTPUT"
    DEFINES="-DCUDA_BACKEND $DEVICE_DEFINE"
    compile_test $PROGRAM $OUTPUT "$DEFINES"
fi

if [ ! -v SKIP_THREAD ]; then
    OUTPUT=$BASE_OUTPUT"_thread"
    echo -e "$ARROW$OUTPUT"
    DEFINES="-DTHREAD_BACKEND $DEVICE_DEFINE"
    compile_test $PROGRAM $OUTPUT "$DEFINES"
fi

#PROGRAM=./pingpong_mpi.cpp
#OUTPUT=mpi-test
#DEFINES="-DNEED_HIP"
#compile_test $PROGRAM $OUTPUT "$DEFINES"

