#!/bin/bash
cd ../build
# Get arguments
ENV=${1:-"obs1"}
INIT_FILE=${2:-"init_guess_mujoco"}
RESULT_NAME=${3:-"result_dbcbs_opt"}
TRIAL=${4:-"000"}

if TRIAL==""
then
    TRIAL="000"
fi
# Shift the first three arguments so "$@" contains only extra flags/args
shift 3

./deps/dynoplan/main_mujoco_opt_simulate \
    --env_file ../stats_db/mujocoquadspayload_${ENV}/${TRIAL}/env.yaml \
    --cfg_file ../configs/opt.yaml \
    --results_path ../stats_db/mujocoquadspayload_${ENV}/${TRIAL}/${RESULT_NAME}.yaml \
    --dynobench_base ../deps/dynoplan/dynobench/ \
    --init_file ../stats_db/mujocoquadspayload_${ENV}/${TRIAL}/${INIT_FILE}.yaml \
    "$@"

trap 'cd ../commands' EXIT