#!/bin/bash

ARROW='\033[0;34m--> \033[0m'

#Stop on fail
set -e

# Switch between Tioga and Tuo modules
if [ $# -eq 0 ]; then
    echo "Running for the MI250X"
    SYSTEM="TIOGA"
else
    echo "Running for the MI300A"
    SYSTEM="TUOLUMNE"
fi

## Compile normal version 
cd ../multi-backend
if [ "$SYSTEM" == "TUOLUMNE" ]; then
    ./compile.sh -f ../benchmark/pingpong_st_db.cpp -C -d
    ./compile.sh -f ../benchmark/pingpong_st.cpp -C -d
else
    ./compile.sh -f ../benchmark/pingpong_st_db.cpp -C
    ./compile.sh -f ../benchmark/pingpong_st.cpp -C
fi
cd ../benchmark


if [ "$SYSTEM" == "TUOLUMNE" ]; then
    module load craype-accel-amd-gfx942
else
    module load craype-accel-amd-gfx90a
fi
## Compile MPI Version

EXEC="../scratch/execs/pingpong_st_db_"$SYSTEM"_mpi_db"
echo -e "$ARROW$EXEC"
set -x
CC -D__HIP_PLATFORM_AMD__ -O3 -g -std=c++20 -x hip -o$EXEC pingpong_mpi_db.cpp -DNEED_HIP
set +x

EXEC="../scratch/execs/pingpong_st_db_"$SYSTEM"_mpi"
echo -e "$ARROW$EXEC"
set -x
CC -D__HIP_PLATFORM_AMD__ -O3 -g -std=c++20 -x hip -o$EXEC pingpong_mpi.cpp -DNEED_HIP
set +x