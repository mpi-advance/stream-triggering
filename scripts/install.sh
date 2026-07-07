#!/bin/bash

CYAN='\033[0;36m'
RESET='\033[0m'

usage() {
    echo "Usage: $0 [-B value] [-m file]"
    echo " -B Build mode: 0 (Debug, default), 1 (Release), 2 (RelWithDebInfo)"
    echo " -m Custom module list file (defaults to <cluster_name>_modules.txt)"
}

# Initialize variables
BUILD_MODE=0
CUSTOM_MODULE_FILE=""
CMAKE_EXTRAS=()

while getopts ":B:m:" opt; do
    case $opt in
        B)
            BUILD_MODE="$OPTARG"
            ;;
        m)
            CUSTOM_MODULE_FILE="$OPTARG"
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

set -e

### 1. Detect Cluster Name
CLUSTER_NAME="${LCSCHEDCLUSTER:-${LMOD_SYSTEM_NAME}}"

if [ -z "$CLUSTER_NAME" ]; then
    echo "Error: Neither LCSCHEDCLUSTER nor LMOD_SYSTEM_NAME environment variables are set."
    exit 1
fi

# Normalize to lowercase for robust matching
CLUSTER_NAME=$(echo "$CLUSTER_NAME" | tr '[:upper:]' '[:lower:]')
echo -e "Detected cluster: ${CYAN}${CLUSTER_NAME}${RESET}"

### 2. Prepare System Specific Variables
if [[ "$CLUSTER_NAME" == *"tioga"* ]]; then
    GPU_ARCH="gfx90a"
elif [[ "$CLUSTER_NAME" == *"tuolumne"* ]]; then
    GPU_ARCH="gfx942"
elif [[ "$CLUSTER_NAME" == *"frontier"* ]]; then
    GPU_ARCH="gfx90a"
    export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
    CMAKE_EXTRAS=( "-DCMAKE_SHARED_LINKER_FLAGS=${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}" )
else
    echo "Error: Unknown or unsupported cluster '$CLUSTER_NAME'."
    exit 1
fi

### 3. Handle Module File
if [ -n "$CUSTOM_MODULE_FILE" ]; then
    MODULE_FILE="$CUSTOM_MODULE_FILE"
else
    MODULE_FILE="${CLUSTER_NAME}_modules.txt"
fi

if [ ! -f "$MODULE_FILE" ]; then
    echo "Error: Module file '$MODULE_FILE' not found."
    exit 1
fi

echo "Reading modules from $MODULE_FILE..."

# Read modules, ignoring comments (#) and empty lines
MODULES_TO_LOAD=$(grep -v '^\s*#' "$MODULE_FILE" | grep -v '^\s*$' || true)

if [ -z "$MODULES_TO_LOAD" ]; then
    echo "Warning: No modules found in $MODULE_FILE."
else
    # Load the modules inline
    module load $MODULES_TO_LOAD
fi

module list

### 4. Dynamically Parse Libfabric Path
# Grep for 'libfabric/' out of the loaded module list
LIBFABRIC_MOD=$(echo "$MODULES_TO_LOAD" | grep -o 'libfabric/[^ ]*' | head -n 1 || true)

if [ -n "$LIBFABRIC_MOD" ]; then
    # Extract just the version number by stripping the prefix
    LIBFABRIC_VER="${LIBFABRIC_MOD#libfabric/}"
    LIBFABRIC_DIR="/opt/cray/libfabric/${LIBFABRIC_VER}"
else
    # Fallback to SYSTEM logic if no explicit libfabric module is loaded
    echo "Notice: 'libfabric' not found in $MODULE_FILE. Defaulting to SYSTEM (/usr/lib64)."
    LIBFABRIC_DIR="/usr/lib64"
fi

echo "Using Libfabric prefix: $LIBFABRIC_DIR"

### 5. Build Configuration
if [ "$BUILD_MODE" -eq 0 ]; then
    MODE="Debug"
elif [ "$BUILD_MODE" -eq 1 ]; then
    MODE="Release"
elif [ "$BUILD_MODE" -eq 2 ]; then
    MODE="RelWithDebInfo"
else
    echo "Invalid build mode specified."
    exit 1
fi

echo -e "Building in ${CYAN}${MODE}${RESET} mode"

DIR_TO_BUILD="build"
if [ -d $DIR_TO_BUILD ]; then
    rm -rf $DIR_TO_BUILD
fi
mkdir $DIR_TO_BUILD && cd $DIR_TO_BUILD

cmake -DUSE_HIP_BACKEND=ON -DUSE_CXI_BACKEND=ON -DLIBFABRIC_PREFIX=${LIBFABRIC_DIR} \
      -DCMAKE_HIP_ARCHITECTURES=${GPU_ARCH} -DCMAKE_INSTALL_PREFIX=${HOME}/apps/foo \
      "${CMAKE_EXTRAS[@]}" -DCMAKE_BUILD_TYPE=$MODE ..

make -j8
make install