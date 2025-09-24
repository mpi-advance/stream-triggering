#!/bin/bash

#Stop on fail
set -e

FINAL_FILE=run.csv
TEMP_FILE=data.txt
#OUT_FILE="flux-f4XTQzfbV6wH.out"
#OUT_FILE="flux-f4XfRR1veTAw.out"
OUT_FILE="flux-f29jbZ6SGPNX.out"

cd testing_dir

rm -f $TEMP_FILE
rm -f $FINAL_FILE

grep "0 is" ../$OUT_FILE > $TEMP_FILE

# Data about where to look in run output
data_index=1
NUM_RUNS=5
POW2_OFFSET=2
NUM_ITERS=100000

# Data for formatting CSV output
COMM_LOC=GPU
COMM_TYPE=ONE_SIDED
API_TYPE=CXI
GRAIN=COARSE

# CXI Coarse
cut -d' ' -f4 $TEMP_FILE | sed -n "${data_index}~${NUM_RUNS}p" | awk -v a=$COMM_LOC -v b=$COMM_TYPE -v c=$API_TYPE -v d=$GRAIN -v e=$POW2_OFFSET -v f=$NUM_ITERS '{print a "," b "," c "," d "," 2^(NR+e) "," f "," $0}' >> $FINAL_FILE
data_index=$((data_index + 1))

# CXI Fine
GRAIN=FINE
cut -d' ' -f4 $TEMP_FILE | sed -n "${data_index}~${NUM_RUNS}p" | awk -v a=$COMM_LOC -v b=$COMM_TYPE -v c=$API_TYPE -v d=$GRAIN -v e=$POW2_OFFSET -v f=$NUM_ITERS '{print a "," b "," c "," d "," 2^(NR+e) "," f "," $0}' >> $FINAL_FILE
data_index=$((data_index + 1))

# HIP
GRAIN=COARSE
API_TYPE=HIP
COMM_TYPE=TWO_SIDED
cut -d' ' -f4 $TEMP_FILE | sed -n "${data_index}~${NUM_RUNS}p" | awk -v a=$COMM_LOC -v b=$COMM_TYPE -v c=$API_TYPE -v d=$GRAIN -v e=$POW2_OFFSET -v f=$NUM_ITERS '{print a "," b "," c "," d "," 2^(NR+e) "," f "," $0}' >> $FINAL_FILE
data_index=$((data_index + 1))

# Thread
API_TYPE=THREAD
cut -d' ' -f4 $TEMP_FILE | sed -n "${data_index}~${NUM_RUNS}p" | awk -v a=$COMM_LOC -v b=$COMM_TYPE -v c=$API_TYPE -v d=$GRAIN -v e=$POW2_OFFSET -v f=$NUM_ITERS '{print a "," b "," c "," d "," 2^(NR+e) "," f "," $0}' >> $FINAL_FILE
data_index=$((data_index + 1))

# MPI
API_TYPE=MPI
COMM_LOC=HOST
cut -d' ' -f4 $TEMP_FILE | sed -n "${data_index}~${NUM_RUNS}p" | awk -v a=$COMM_LOC -v b=$COMM_TYPE -v c=$API_TYPE -v d=$GRAIN -v e=$POW2_OFFSET -v f=$NUM_ITERS '{print a "," b "," c "," d "," 2^(NR+e) "," f "," $0}' >> $FINAL_FILE
data_index=$((data_index + 1))
