#!/bin/bash
#flux: --nodes=16
#flux: --nslots=64
#flux: --time-limit=5m
#flux: --queue=pdebug
#flux: --exclusive
#flux: --output=../scratch/flux/{{jobid}}.out

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
TEST_NAME=halo
#TEST_NAME=rsend
TIME=2m
NUM_ITERS=50
BUFF_SIZE=50
NODES=16
PPN=4

cd scratch/tmp/

# Print out variables in run file just for tracking
HOSTNAMES_FILE="a-hostnames.tmp"
VAR_MOD_FILE="a-var-mod.tmp"
echo "$TEST_NAME,$SYSTEM" >> $VAR_MOD_FILE
module list >> $VAR_MOD_FILE 2>&1
srun --output=$HOSTNAMES_FILE hostname

# Function for running test
run_test()(
    RUN_FILE="$1.tmp"
    STRING="Test: $1 $NUM_ITERS $BUFF_SIZE"
    flux run --time-limit=$TIME --nodes=$NODES --tasks-per-node=$PPN --output=$RUN_FILE "../execs/${TEST_NAME}_${SYSTEM}_$1" $NUM_ITERS $BUFF_SIZE
    sed -i "1i$STRING" $RUN_FILE
)

run_test "cxi-coarse"
run_test "cxi-fine"

export MPICH_GPU_SUPPORT_ENABLED=1
#run_test "hip"
#run_test "thread"
unset MPICH_GPU_SUPPORT_ENABLED

# While slurm has append to file, flux does not. So we have to 
# manage temporary output files.
cat *.tmp >> $TARGET
rm -f *.tmp
