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

## Get the modules for the system
SYSTEM="${LCSCHEDCLUSTER}"
MOD_FILE="module_sets/${SYSTEM}"
module load $(cat "$MOD_FILE")

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
export MPICH_GPU_SUPPORT_ENABLED=1

# Settings related to individual tests
TEST_NAME=pingpong
TIME=3m
START_EXP=3
END_EXP=28
NUM_ITERS=100000

# Add hostnames to file
srun --nodes=$NODES --ntasks-per-node=1 --output=$TARGET hostname
# Save modules
VAR_MOD_FILE=$TARGET
module list >> $VAR_MOD_FILE 2>&1

# Function for running test
run_test()(
    EXEC="./scratch/execs/${TEST_NAME}_${1}_${SYSTEM}"
    TEST="$1"
    if [ -n "$2" ]; then
        EXEC="${EXEC}_$2"
        TEST="${TEST}-$2"
    fi
    echo "Test: ${TEST} $NUM_ITERS $BUFF_SIZE" >> $TARGET
    flux run -N$NODES --tasks-per-node=$PPN --time-limit=$TIME \
            --output="$TARGET" -o output.mode=append           \
             "${EXEC}" $NUM_ITERS $BUFF_SIZE
)

run_db_test()(
    EXEC="./scratch/execs/${TEST_NAME}_${1}_db_${SYSTEM}"
    TEST="${1}"
    if [ -n "$2" ]; then
        EXEC="${EXEC}_$2"
        TEST="${TEST}-${2}"
    fi
    TEST="${TEST}_db"
    echo "Test: ${TEST} $NUM_ITERS $BUFF_SIZE" >> $TARGET
    flux run -N$NODES --tasks-per-node=$PPN --time-limit=$TIME \
            --output="$TARGET" -o output.mode=append           \
             "${EXEC}" $NUM_ITERS $BUFF_SIZE
)

run_tests()
(
    run_test "$@"
    run_db_test "$@"
)

for (( exp=START_EXP; exp<=END_EXP; exp++ )); do
    BUFF_SIZE=$((2 ** $exp))

    if [ $BUFF_SIZE -ge 16777216 ]; then
        NUM_ITERS=1000
    elif [ $BUFF_SIZE -ge 1048576 ]; then
        NUM_ITERS=10000
    fi

    echo "Starting round: $NUM_ITERS $BUFF_SIZE"

    run_tests "st" "cxi-coarse"
    #run_tests "st" "cxi-fine"

    #run_tests "hip"
    #run_tests "thread"
    run_tests "mpi"
    run_test "ipc"
    export HSA_ENABLE_SDMA=0
    run_test "ipc"
    export HSA_ENABLE_SDMA=1

done
