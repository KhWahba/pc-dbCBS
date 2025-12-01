
PROBLEM_NAME="mujocoquadspayload_$1"

ENV_FILE="../deps/dynoplan/dynobench/envs/mujoco/$PROBLEM_NAME.yaml"
PAYLOAD_FILE="../stats_db/$PROBLEM_NAME/000/result_dbcbs_mujoco.yaml"
RESULT_FILE="../stats_db/$PROBLEM_NAME/000/result_dbcbs.yaml"
OUTPUT_FILE="../stats_db/${PROBLEM_NAME}/000/init_guess_mujoco.html"

python3 ../scripts/mesh_visualizer.py --env "$ENV_FILE" --payload "$PAYLOAD_FILE" --result "$RESULT_FILE" --video "$OUTPUT_FILE"
echo "Visualization saved to $OUTPUT_FILE"