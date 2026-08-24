#!/bin/bash
#flux: --nodes=1
#flux: --nslots=1
#flux: --time-limit=5m
#flux: --queue=pdebug
#flux: --exclusive
#flux: --output=../scratch/flux/{{jobid}}.out

### 0. Global Options / Env vars

# Settings related to individual tests
TEST_NAME=self_send
TIME=3m
NUM_ITERS=50
BUFF_SIZE=10
NODES=1
PPN=1

# set -e
# ulimit -c unlimited
USE_ROCPROF=false

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/apps/stream-trigger/lib
#export HSA_USE_SVM=0
export HSA_XNACK=1
#export MPICH_ASYNC_PROGRESS=1
export MPICH_GPU_SUPPORT_ENABLED=1

### 0.5 Go up a level to start
cd ..

### 1. Robust Module Loading
SYSTEM="${LCSCHEDCLUSTER}"
MOD_FILE="../install_setup/${SYSTEM}_modules.txt"
#MOD_FILE="../install_setup/tuolumne_rocm_7.14modules.txt"

if [ -f "$MOD_FILE" ]; then
    MODULES=$(grep -v '^\s*#' "$MOD_FILE" | grep -v '^\s*$' || true)
    module load $MODULES
else
    echo "Warning: Module file not found at $MOD_FILE"
fi

### 2. Output File Setup (Simplified)
USER_BASE="$HOME/git/stream-triggering/tests/scratch"
mkdir -p "$USER_BASE/output" # Ensure directory exists
TARGET="$USER_BASE/output/$SYSTEM-$(date +%m-%d-%H%M%S).out"
touch "$TARGET"
echo "Outputting to: $TARGET"

cd scratch/tmp/

# Record hostnames and modules for debugging
srun --nodes=2 --ntasks-per-node=1 --output="$TARGET" hostname
module list >> "$TARGET" 2>&1

# Function for running test
run_test()(
    echo "Test: $1 $NUM_ITERS $BUFF_SIZE" >> "$TARGET"

    # Handle Rocprof Profiler Toggle 
    local profiler_cmd=""
    if [[ "$USE_ROCPROF" == "true" ]]; then
        profiler_cmd="rocprofv3 --sys-trace --output-format pftrace --"
    fi

    flux run --time-limit=$TIME --nodes=$NODES --tasks-per-node=$PPN \
             --output="$TARGET" -o output.mode=append \
             ${profiler_cmd} "../execs/${TEST_NAME}_${SYSTEM}_$1" $NUM_ITERS $BUFF_SIZE
)

run_test "cxi-coarse"
#run_test "cxi-fine"

#run_test "hip"
#run_test "thread"