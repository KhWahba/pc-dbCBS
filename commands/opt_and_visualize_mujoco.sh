#!/bin/bash
cd ../build
# Get arguments
PROBLEM_NAME=$1
INIT_FILE=$2
RESULT_NAME=$3
TRIAL=$4

# Shift the first three arguments so "$@" contains only extra flags/args
shift 3

./deps/dynoplan/main_mujoco_opt_simulate \
    --env_file ../stats_db/${PROBLEM_NAME}/${TRIAL}/env.yaml \
    --cfg_file ../configs/opt.yaml \
    --results_path ../stats_db/${PROBLEM_NAME}/${TRIAL}/${RESULT_NAME}.yaml \
    --dynobench_base ../deps/dynoplan/dynobench/ \
    --init_file ../stats_db/${PROBLEM_NAME}/${TRIAL}/${INIT_FILE}.yaml \
    "$@"

trap 'cd ../commands' EXIT