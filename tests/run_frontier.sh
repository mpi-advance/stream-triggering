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
#   ST_LAUNCHER_LAYOUT  on-node / off-node (default; see launch.sh). Set it
#                       empty to leave placement to the launcher.
#   CTEST_ARGS          Extra ctest arguments (e.g. "-R halo")
#   ST_TEST_SIZE        Elements (ints) per buffer (CXI's credit cutoff is 8192)
#   ST_TEST_ITERS       Iterations, for tests that loop
#   ST_TEST_VERBOSE     1 for the tests' informational output

ST_ROOT=$(git rev-parse --show-toplevel)
source "${ST_ROOT}/install_setup/st_env.sh"

# One rank per node, to match the two-node allocation
export ST_LAUNCHER_LAYOUT=${ST_LAUNCHER_LAYOUT-off-node}

srun --ntasks-per-node=1 hostname
st_run_ctest "${1:-${BACKEND}}"
