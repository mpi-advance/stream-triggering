# Setup for Tioga

## Building

```bash
module load craype-accel-amd-gfx90a
module load rocm

INSTALL_DIR="" # Fill in path to where you want to install library
LIBFABRIC_DIR="" # Fill in path to where libfabric is
cmake -DUSE_IMPLEMENTATION=CXI -DLIBFABRIC_PREFIX=$LIBFABRIC_DIR -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=CC ..
```

## Running

Script to compile:
```bash
module load craype-accel-amd-gfx90a
module load rocm

ST_PATH="" # Fill in path to where library is installed
ST_LIB_PATH=$ST_PATH/lib
ST_INC_PATH=$ST_PATH/include

PROGRAM=cxi_simple.cpp

CC -D__HIP_PLATFORM_AMD__ -O3 -g -std=c++20 -x hip $PROGRAM -I$ST_INC_PATH -L$ST_LIB_PATH -o test -lstream-triggering
```

Script to run (slurm stuff is converted to flux automatically):
```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --partition=pdebug

module load craype-accel-amd-gfx90a
module load rocm

# NEED THIS FOR CXI STUFF!
export HSA_USE_SVM=0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"" # Fill in quotes with the final value of $ST_LIB_PATH from compiling

srun ./test
```

Submit job normally:
```bash
sbatch ./run_script.sh
```

