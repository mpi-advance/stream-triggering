# Setup for Tioga

## Building
Example scipts can be found in the `install` folder. Examples of compiling against and using the library (on Tioga) can be found in the `test` folder.

The rest of the examples below more general examples that may be incomplete, so don't just copy paste them.

```bash
module load craype-accel-amd-gfx90a
module load rocm

INSTALL_DIR="" # Fill in path to where you want to install library
LIBFABRIC_DIR="" # Fill in path to where libfabric is

cmake -DUSE_IMPLEMENTATION=CXI -DLIBFABRIC_PREFIX=$LIBFABRIC_DIR -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_C_COMPILER=hipcc -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ..
```

Build using just HIP commands (and MPI). Currently, the `GPU_MEM_OPS` implementation also needs `USE_GPU_TYPE` to be specified (`AMD` or `NVIDIA`). `Release` mode to turn of debug print outs.

```bash
module load craype-accel-amd-gfx90a
module load rocm

INSTALL_DIR="" # Fill in path to where you want to install library
cmake -DCMAKE_BUILD_TYPE=RELEASE -DUSE_IMPLEMENTATION=GPU_MEM_OPS -DUSE_GPU_TYPE=AMD -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=CC -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ..
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

