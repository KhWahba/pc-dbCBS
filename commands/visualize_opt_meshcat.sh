
PROBLEM_NAME="mujocoquadspayload_$1"

ENV_FILE="../stats_db/$PROBLEM_NAME/000/env.yaml"
RESULT_FILE="../stats_db/$PROBLEM_NAME/000/$2.yaml"
OUTPUT_FILE="../stats_db/$PROBLEM_NAME/000/$2.html"

python3 ../scripts/visualize_mujoco.py --env "$ENV_FILE" --result "$RESULT_FILE" --output "$OUTPUT_FILE"
echo "Visualization saved to $OUTPUT_FILE"