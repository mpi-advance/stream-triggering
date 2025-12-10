#!/bin/bash

ARROW='\033[0;34m--> \033[0m'

#Stop on fail
set -e

# Switch between Tioga and Tuo modules
if [ $# -eq 0 ]; then
    echo "Running for Tioga"
    SYSTEM="TIOGA"
    VERSION=0
elif [ $1 -eq 1 ]; then
    echo "Running for Tuolumne"
    SYSTEM="TUOLUMNE"
    VERSION=1
elif [ $1 -eq 2 ]; then
    echo "Running for Frontier"
    SYSTEM="FRONTIER"
    VERSION=2
fi

if [ "$SYSTEM" == "FRONTIER" ]; then
    ST_PATH=/ccs/home/dschafer/apps/stream_trigger
else
    ST_PATH=/g/g16/derek/apps/stream_trigger
fi


## Compile normal version 
cd ../multi-backend
./compile.sh -f ../benchmark/pingpong_st_db.cpp -C -d $VERSION -S $ST_PATH
./compile.sh -f ../benchmark/pingpong_st.cpp -C -d $VERSION -S $ST_PATH
cd ../benchmark


if [ "$SYSTEM" == "TUOLUMNE" ]; then
    module load craype-accel-amd-gfx942
else
    module load craype-accel-amd-gfx90a
fi
module load rocm
## Compile MPI Version

EXEC="../scratch/execs/pingpong_st_db_"$SYSTEM"_mpi"
echo -e "$ARROW$EXEC"
set -x
CC -D__HIP_PLATFORM_AMD__ -O3 -g -std=c++20 -x hip -o$EXEC pingpong_mpi_db.cpp -DNEED_HIP
set +x

EXEC="../scratch/execs/pingpong_st_"$SYSTEM"_mpi"
echo -e "$ARROW$EXEC"
set -x
CC -D__HIP_PLATFORM_AMD__ -O3 -g -std=c++20 -x hip -o$EXEC pingpong_mpi.cpp -DNEED_HIP
set +x
