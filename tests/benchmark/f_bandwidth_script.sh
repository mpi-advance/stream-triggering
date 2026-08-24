#!/bin/bash
#SBATCH --nodes=2
#SBATCH --time=00:05:00
#SBATCH --partition=batch
#SBATCH --qos=debug
#SBATCH --account=csc698
#SBATCH --output=../scratch/flux/%j.out
#SBATCH --exclusive

### 0. Global Options / Env vars

# set -e
# ulimit -c unlimited
USE_ROCPROF=false

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/apps/stream-trigger/lib
#export HSA_USE_SVM=0
export HSA_XNACK=1
#export MPICH_ASYNC_PROGRESS=1
export MPICH_GPU_SUPPORT_ENABLED=1

TEST_NAME="pingpong"
TIME="00:03:00"
START_EXP=15
END_EXP=15

### 0.5 Go up a level to start
cd ..

### 1. Robust Module Loading
SYSTEM="${LMOD_SYSTEM_NAME}"
MOD_FILE="../install_setup/${SYSTEM}_modules.txt"
#MOD_FILE="../install_setup/tuolumne_rocm7_modules.txt"

if [ -f "$MOD_FILE" ]; then
    MODULES=$(grep -v '^\s*#' "$MOD_FILE" | grep -v '^\s*$' || true)
    module load $MODULES
else
    echo "Warning: Module file not found at $MOD_FILE"
fi

### 1.5 Cray-specific, frontier-specific module adjusting:
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

### 2. Output File Setup (Simplified)
USER_BASE="$HOME/git/stream-triggering/tests/scratch"
mkdir -p "$USER_BASE/output" # Ensure directory exists
TARGET="$USER_BASE/output/$SYSTEM-$(date +%m-%d-%H%M%S).out"
touch "$TARGET"
echo "Outputting to: $TARGET"

# Record hostnames and modules for debugging
srun --nodes=2 --ntasks-per-node=1 --output="$TARGET" hostname
module list >> "$TARGET" 2>&1

### 3. Core Execution Function
run_test_instance() {
    local base_test="$1"
    local variant="$2"
    local mode="$3" # "sb" or "db"
    local topo_label="$4"
    local n_nodes="$5"
    local ppn="$6"
    local custom_label="$7"

    TLES=$((1024 / $ppn))
    
    # Determine executable name
    local exec_name="${base_test}"
    if [[ "$mode" == "db" ]]; then
        exec_name="${base_test}_db"
    fi
    
    local exec_path="./scratch/execs/${TEST_NAME}_${exec_name}_${SYSTEM}"
    local test_label="${base_test}"

    # Append variants if they exist (e.g., cxi-coarse)
    if [ -n "$variant" ]; then
        exec_path="${exec_path}_${variant}"
        test_label="${test_label}-${variant}"
        if [ -n "$custom_label" ]; then
            test_label="${test_label}-${custom_label}"
        fi
    fi

    echo "Test: ${test_label} ${mode} ${topo_label} $NUM_ITERS $BUFF_SIZE" >> "$TARGET"

    # Handle Rocprof Profiler Toggle 
    local profiler_cmd=""
    if [[ "$USE_ROCPROF" == "true" ]]; then
        profiler_cmd="rocprofv3 --sys-trace --output-format pftrace --"
    fi

    srun --network=single_node_vni,job_vni,def_tles=$TLES -N "$n_nodes" --ntasks-per-node="$ppn" \
         --time="$TIME" --output="$TARGET" \
         "${exec_path}" "$NUM_ITERS" "$BUFF_SIZE"
}

### 4. Wrapper Function for DB/SB and Topology
run_tests() {
    local base_test="$1"
    local variant="$2"
    local custom_label="$3"
    
    local run_sb="${do_sb:-true}"
    local run_db="${do_db:-true}"
    local run_on="${do_on_node:-true}"
    local run_off="${do_off_node:-true}"

    # Loop through execution topologies
    for topo in "ON-NODE" "OFF-NODE"; do
        
        # Check explicit topology overrides
        if [[ "$topo" == "ON-NODE" ]]; then
            [[ "$run_on" != "true" ]] && continue
            local n=1
            local p=2
        else
            [[ "$run_off" != "true" ]] && continue
            local n=2
            local p=1
        fi

        if [[ "$run_sb" == "true" ]]; then
            run_test_instance "$base_test" "$variant" "sb" "$topo" "$n" "$p" "$custom_label"
        fi

        if [[ "$run_db" == "true" ]]; then
            run_test_instance "$base_test" "$variant" "db" "$topo" "$n" "$p" "$custom_label"
        fi
    done
}

### 5. Main Execution Loop
for (( exp=START_EXP; exp<=END_EXP; exp++ )); do
    BUFF_SIZE=$((2 ** exp))

    if [ "$BUFF_SIZE" -ge 16777216 ]; then
        NUM_ITERS=500
    #elif [ "$BUFF_SIZE" -ge 1048576 ]; then
    #    NUM_ITERS=5000
    else
        NUM_ITERS=10
    fi

    echo "Starting round: $NUM_ITERS $BUFF_SIZE"

    run_tests "st" "cxi-coarse"
    #run_tests "mpi"
    export MPIA_ST_DISABLE_CREDIT=1
    #do_on_node=false do_db=false run_tests "st" "cxi-coarse" "no-credit"
    unset MPIA_ST_DISABLE_CREDIT
    #do_off_node=false do_db=false run_tests "ipc" 
done
