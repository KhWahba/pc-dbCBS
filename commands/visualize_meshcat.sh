
PROBLEM_NAME="$1"

ENV_FILE="../stats_db/$PROBLEM_NAME/000/env.yaml"
RESULT_FILE="../stats_db/$PROBLEM_NAME/000/$2.yaml"
OUTPUT_FILE="../stats_db/$PROBLEM_NAME/000/$2.html"
PAYLOAD_FLAG="$3"

python3 ../scripts/visualize_mujoco.py --env "$ENV_FILE" --result "$RESULT_FILE" --output "$OUTPUT_FILE" $PAYLOAD_FLAG
echo "Visualization saved to $OUTPUT_FILE"