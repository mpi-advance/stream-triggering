#!/bin/bash

#Stop on fail
set -e


ST_PATH=${HOME}/apps/stream_trigger

cd ..
python3 compile.py -f benchmark/pingpong_st_db.cpp -C -T -S $ST_PATH
python3 compile.py -f benchmark/pingpong_st.cpp -C -T -S $ST_PATH
python3 compile.py -f benchmark/pingpong_mpi_db.cpp 
python3 compile.py -f benchmark/pingpong_mpi.cpp
python3 compile.py -f benchmark/pingpong_ipc.cpp