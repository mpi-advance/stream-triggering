#!/bin/bash
# Shared environment setup for building and running stream-triggering.
# Source this file (don't run it), then call the st_* functions below.

ST_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

# Print the lowercased cluster name, or nothing when not on a known scheduler.
st_detect_cluster() {
    local cluster="${LCSCHEDCLUSTER:-${LMOD_SYSTEM_NAME}}"
    echo "$cluster" | tr '[:upper:]' '[:lower:]'
}

# Print the module file recorded by install.sh in a build directory, if any.
st_build_module_file() {
    local build_dir="$1"
    if [ -f "${build_dir}/st_modules.path" ]; then
        cat "${build_dir}/st_modules.path"
    fi
}

# Load the modules listed in a file, skipping comments and blank lines.
# Usage: st_load_modules [-m file]  (defaults to <cluster>_modules.txt)
# Exports ST_MODULE_FILE and ST_MODULES_LOADED.
st_load_modules() {
    local module_file=""
    local OPTIND opt
    while getopts ":m:" opt; do
        case $opt in
            m) module_file="$OPTARG" ;;
            *) echo "Usage: st_load_modules [-m file]"; return 1 ;;
        esac
    done

    if [ -z "$module_file" ]; then
        local cluster
        cluster=$(st_detect_cluster)
        if [ -z "$cluster" ]; then
            echo "Notice: No cluster detected, not loading any modules."
            return 0
        fi
        module_file="${ST_ROOT}/install_setup/${cluster}_modules.txt"
    fi

    if [ ! -f "$module_file" ]; then
        echo "Error: Module file '$module_file' not found."
        return 1
    fi

    if ! type module &> /dev/null; then
        echo "Notice: No 'module' command available, not loading $module_file."
        return 0
    fi

    echo "Reading modules from $module_file..."
    export ST_MODULE_FILE=$(realpath "$module_file")
    export ST_MODULES_LOADED=$(grep -v '^\s*#' "$module_file" | grep -v '^\s*$' || true)

    if [ -z "$ST_MODULES_LOADED" ]; then
        echo "Warning: No modules found in $module_file."
    else
        module load $ST_MODULES_LOADED
    fi

    module list
}

# Per-cluster runtime exports, including the default test launcher (ST_LAUNCHER).
# Off-cluster, ST_LAUNCHER is left alone and tests/launch.sh falls back to the
# MPI launcher CMake found.
st_cluster_env() {
    local cluster
    cluster=$(st_detect_cluster)

    if [[ "$cluster" == *"tuolumne"* ]]; then
        export HSA_XNACK=1
        export ST_LAUNCHER="${ST_LAUNCHER:-flux run}"
    elif [[ "$cluster" == *"frontier"* ]]; then
        export HSA_XNACK=1
        export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
        export ST_LAUNCHER="${ST_LAUNCHER:-srun --network=job_vni,res_tles=1024 --gpus-per-task=1}"
    fi
}

# Load a build's modules and environment, then run its CTest suite. JUnit and log
# output go under tests/scratch/. Usage: st_run_ctest [backend label]
# Uses BUILD_DIR, MOD_FILE and CTEST_ARGS from the environment (see tests/run_*.sh).
st_run_ctest() {
    local backend="$1"
    local build_dir="${BUILD_DIR:-${ST_ROOT}/build}"
    local mod_file="${MOD_FILE:-$(st_build_module_file "$build_dir")}"

    st_load_modules ${mod_file:+-m "$mod_file"} || return 1
    st_cluster_env

    local scratch="${ST_ROOT}/tests/scratch"
    mkdir -p "${scratch}/output" "${scratch}/junit"
    local cluster run_name
    cluster=$(st_detect_cluster)
    run_name="${cluster:-local}-${backend:-all}-$(date +%m-%d-%H%M%S)"
    echo "Running: ${run_name}"

    ctest --test-dir "$build_dir" ${backend:+-L "$backend"} --timeout 180 \
          --output-on-failure --output-junit "${scratch}/junit/${run_name}.xml" \
          --output-log "${scratch}/output/${run_name}.log" ${CTEST_ARGS}
}
