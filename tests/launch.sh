#!/bin/bash
# Launch one test under MPI: launch.sh <ranks> <exe> [args...]
# CTest runs every test through this script.
#
# Environment:
#   ST_LAUNCHER         Launcher command (e.g. "flux run", "srun ..."). Set by
#                       st_cluster_env on known clusters; otherwise the MPI
#                       launcher CMake found (ST_DEFAULT_LAUNCHER) is used.
#   ST_LAUNCHER_LAYOUT  on-node (all ranks on one node) or off-node (one rank
#                       per node). Unset leaves placement to the launcher.
#                       Only flux and srun support it.
#   ST_LAUNCHER_ARGS    Extra arguments placed before the executable (e.g. a
#                       profiler wrapper).

if [ $# -lt 2 ]; then
    echo "Usage: $0 <ranks> <exe> [args...]"
    exit 1
fi

RANKS="$1"
shift

LAUNCHER=( ${ST_LAUNCHER:-${ST_DEFAULT_LAUNCHER:-mpirun}} )
LAUNCHER_NAME=$(basename "${LAUNCHER[0]}")

LAYOUT_ARGS=()
case "$LAUNCHER_NAME" in
    flux|srun)
        RANK_ARGS=( -n "$RANKS" )
        if [ "$ST_LAUNCHER_LAYOUT" == "on-node" ]; then
            LAYOUT_ARGS=( -N 1 )
        elif [ "$ST_LAUNCHER_LAYOUT" == "off-node" ]; then
            if [ "$LAUNCHER_NAME" == "flux" ]; then
                LAYOUT_ARGS=( -N "$RANKS" --tasks-per-node=1 )
            else
                LAYOUT_ARGS=( -N "$RANKS" --ntasks-per-node=1 )
            fi
        fi
        ;;
    *)
        RANK_ARGS=( ${ST_DEFAULT_RANK_FLAG:--np} "$RANKS" )
        if [ -n "$ST_LAUNCHER_LAYOUT" ]; then
            echo "Warning: ST_LAUNCHER_LAYOUT is ignored for $LAUNCHER_NAME."
        fi
        ;;
esac

set -x
exec "${LAUNCHER[@]}" "${RANK_ARGS[@]}" "${LAYOUT_ARGS[@]}" ${ST_LAUNCHER_ARGS} "$@"
