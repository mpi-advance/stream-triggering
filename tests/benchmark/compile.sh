#!/bin/bash

usage() {
    echo "Usage: $0 [-m file]"
    echo " -m Custom module list file to pass (defaults to passing nothing )"
}

while getopts ":m:" opt; do
    case $opt in
        m)
            CUSTOM_MODULE_FILE="$OPTARG"
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

#Stop on fail
set -e

module_command=""
if [ -n "$CUSTOM_MODULE_FILE" ]; then
    module_command="-m ${CUSTOM_MODULE_FILE}"
fi

ST_PATH=${HOME}/apps/stream-trigger

cd ..
python3 compile.py -f benchmark/pingpong_st_db.cpp -C -T -S $ST_PATH ${module_command}
python3 compile.py -f benchmark/pingpong_st.cpp -C -T -S $ST_PATH ${module_command}
python3 compile.py -f benchmark/pingpong_mpi_db.cpp ${module_command}
python3 compile.py -f benchmark/pingpong_mpi.cpp ${module_command}
python3 compile.py -f benchmark/pingpong_ipc.cpp ${module_command}