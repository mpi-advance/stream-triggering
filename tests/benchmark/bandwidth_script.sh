#!/bin/bash
#flux: --nodes=1
#flux: --nslots=2
#flux: --time=1h
#flux: --queue=pbatch
#flux: --gpus-per-slot=1
#flux: --output=../scratch/flux/{{jobid}}.out
#flux: --exclusive
#flux: --env=NODES={{nnodes}}
PPN=2

# Debugging options
#set -e
#ulimit -c unlimited
## Go up on directory to tests folder
cd ..

# Switch between Tioga and Tuo modules
if [ $# -eq 0 ]; then
    echo "Running for the MI250X"
    module load craype-accel-amd-gfx90a
    SYSTEM="TIOGA"
else
    echo "Running for the MI300A"
    module load craype-accel-amd-gfx942
    SYSTEM="TUOLUMNE"
fi

module load rocm

#Control output
USER_BASE="$HOME/git/stream-triggering/tests/scratch"
FILENAME_BASE="$USER_BASE/output/$SYSTEM-$(date +%m-%d)"
COUNT=1
TARGET="${FILENAME_BASE}-${COUNT}.out"

while [[ -e $TARGET ]]; do
    ((COUNT++))
    TARGET="${FILENAME_BASE}-${COUNT}.out"
done

touch "$TARGET"
echo $TARGET

# Any extra environment variables we need
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/apps/stream_trigger/lib
#export HSA_USE_SVM=0
export HSA_XNACK=1
#export MPICH_ASYNC_PROGRESS=1

# Settings related to individual tests
TEST_NAME=pingpong_st
TIME=3m
START_EXP=3
END_EXP=28
NUM_ITERS=100000

cd scratch/tmp/

# Print out variables in run file just for tracking
HOSTNAMES_FILE="a-hostnames.tmp"
VAR_MOD_FILE="a-var-mod.tmp"
echo "$START_EXP,$END_EXP,$SYSTEM" >> $VAR_MOD_FILE
module list >> $VAR_MOD_FILE 2>&1
srun --output=$HOSTNAMES_FILE hostname

# Function for running test
run_test()(
    RUN_FILE="$1.tmp"
    STRING="Test: $1 $NUM_ITERS $BUFF_SIZE"
    flux run -N$NODES --tasks-per-node=$PPN --output="$RUN_FILE" --time-limit=$TIME "../execs/${TEST_NAME}_${SYSTEM}_$1" $NUM_ITERS $BUFF_SIZE
    sed -i "1i$STRING" $RUN_FILE
)

run_db_test()(
    RUN_FILE="${1}_db.tmp"
    STRING="Test: ${1}_db $NUM_ITERS $BUFF_SIZE"
    flux run -N$NODES --tasks-per-node=$PPN --output="$RUN_FILE" --time-limit=$TIME "../execs/${TEST_NAME}_db_${SYSTEM}_$1" $NUM_ITERS $BUFF_SIZE
    sed -i "1i$STRING" $RUN_FILE
)

run_tests()
(
    run_test $1
    run_db_test $1
)

for (( exp=START_EXP; exp<=END_EXP; exp++ )); do
    BUFF_SIZE=$((2 ** $exp))

    if [ $BUFF_SIZE -ge 16777216 ]; then
        NUM_ITERS=1000
    elif [ $BUFF_SIZE -ge 1048576 ]; then
        NUM_ITERS=10000
    fi

    echo "Starting round: $NUM_ITERS $BUFF_SIZE"

    run_tests "cxi-coarse"
    run_tests "cxi-fine"

    export MPICH_GPU_SUPPORT_ENABLED=1
    run_tests "hip"
    #run_tests "thread"
    run_tests "mpi"
    unset MPICH_GPU_SUPPORT_ENABLED

    # While slurm has append to file, flux does not. So we have to 
    # manage temporary output files.
    cat *.tmp >> $TARGET
    rm -f *.tmp
done
