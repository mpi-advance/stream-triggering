#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:09:00
#SBATCH --partition=pdebug

set -e

module load craype-accel-amd-gfx90a
module load rocm/6.3.1

#export HSA_USE_SVM=0
export HSA_XNACK=1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/g/g16/derek/apps/stream_trigger/lib
ulimit -c unlimited

#export MPICH_ASYNC_PROGRESS=1
#export MPICH_GPU_SUPPORT_ENABLED=1
#echo $HIP_HOST_COHERENT
#echo $hipHostMallocCoherent
#echo $hipHostMallocNonCoherent
#echo $hipHostMallocMapped

cd testing_dir/
echo "Starting CXI Coarse Test"
srun --time=00:02:00 ./cxi-coarse 10000 100000

echo "Starting CXI Fine Test"
srun --time=00:02:00 ./cxi-fine 10000 100000

echo "Starting HIP Test"

export MPICH_GPU_SUPPORT_ENABLED=1
srun --time=00:02:00 ./hip-test 10000 100000

echo "Starting Thread Test"
srun --time=00:02:00 ./thread-test 10000 100000
