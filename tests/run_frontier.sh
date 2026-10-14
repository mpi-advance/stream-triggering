#!/bin/bash
#SBATCH --nodes=2
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --qos=debug
#SBATCH --account=csc698
#SBATCH --exclusive
#SBATCH --output=scratch/flux/%j.out

# Runs the CTest suite on Frontier. Submit from tests/ (after `mkdir -p scratch/flux`):
#   sbatch run_frontier.sh [backend]
# backend is a CTest label (cxi-coarse, cxi-fine, hip, thread); omit it to run
# every variant. Other settings come from the environment (sbatch --export=ALL,...):
#   BUILD_DIR           Build directory with the tests (default: <repo>/build)
#   MOD_FILE            Module file (default: the one install.sh recorded)
#   ST_LAUNCHER_LAYOUT  on-node / off-node (see launch.sh)
#   CTEST_ARGS          Extra ctest arguments (e.g. "-R halo")

ST_ROOT=$(git rev-parse --show-toplevel)
source "${ST_ROOT}/install_setup/st_env.sh"

BACKEND="${1:-${BACKEND}}"
BUILD_DIR="${BUILD_DIR:-${ST_ROOT}/build}"
MOD_FILE="${MOD_FILE:-$(st_build_module_file "$BUILD_DIR")}"

st_load_modules ${MOD_FILE:+-m "$MOD_FILE"}
st_cluster_env

SCRATCH="${ST_ROOT}/tests/scratch"
mkdir -p "${SCRATCH}/output" "${SCRATCH}/junit"
RUN_NAME="$(st_detect_cluster)-${BACKEND:-all}-$(date +%m-%d-%H%M%S)"

echo "Running: ${RUN_NAME}"
srun --ntasks-per-node=1 hostname

ctest --test-dir "$BUILD_DIR" ${BACKEND:+-L "$BACKEND"} --timeout 180 --output-on-failure \
      --output-junit "${SCRATCH}/junit/${RUN_NAME}.xml" \
      --output-log "${SCRATCH}/output/${RUN_NAME}.log" ${CTEST_ARGS}
