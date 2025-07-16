#!/bin/bash

# Set base paths
BASE_DIR="/anycapeval_audio"
SCRIPT_DIR="/anycapeval_audio/eval"  # Please replace with actual script directory
OUTPUT_DIR="${BASE_DIR}/output"
TEST_REFERENCE="${BASE_DIR}/anycapeval_audio_ref.jsonl"

EXP_ID="code_test"  
OUTPUT_FOLDER="${OUTPUT_DIR}/${EXP_ID}/"

# Create output directory
mkdir -p ${OUTPUT_FOLDER}

# Set output paths
OUTPUT_PATH_CONTENT="/anycapeval_audio/eval/input/content_input/test.jsonl"
OUTPUT_PATH_STYLE="/anycapeval_audio/eval/input/style_input/test.jsonl"

# Set evaluation result paths
EVAL_CONTENT="${OUTPUT_FOLDER}/eval_content.jsonl"
EVAL_STYLE="${OUTPUT_FOLDER}/eval_style.jsonl"

# API configuration
API_KEY="YOUR_API_KEY_HERE"

# Other fixed paths
AUDIO_DIR="${BASE_DIR}/test_audio_data"
CONTENT_TEMPLATE_PATH="${BASE_DIR}/instruction_content.txt"
STYLE_TEMPLATE_PATH="${BASE_DIR}/instruction_style.txt"
EVAL_CONTENT_MODEL=""
EVAL_STYLE_MODEL=""

echo "=================== Starting Evaluation Process ==================="
echo "Experiment ID: ${EXP_ID}"
echo "Output Directory: ${OUTPUT_FOLDER}"


# Check if generation was successful
if [ ! -f "${OUTPUT_PATH_CONTENT}" ] || [ ! -f "${OUTPUT_PATH_STYLE}" ]; then
    echo "Error: Response file generation failed, please check logs"
    exit 1
fi

echo "=================== Step 1: Evaluating Content Quality ==================="
python ${SCRIPT_DIR}/anycapeval_audio_content.py \
    --api_key ${API_KEY} \
    --model ${EVAL_CONTENT_MODEL} \
    --data_path ${OUTPUT_PATH_CONTENT} \
    --content_template_path ${CONTENT_TEMPLATE_PATH} \
    --output_path ${EVAL_CONTENT} \
    --test_data_path ${TEST_REFERENCE}

echo "=================== Step 2: Evaluating Style ==================="
python ${SCRIPT_DIR}/anycapeval_audio_style.py \
    --api_key ${API_KEY} \
    --model ${EVAL_STYLE_MODEL} \
    --data_path ${OUTPUT_PATH_STYLE} \
    --prompt_template_path ${STYLE_TEMPLATE_PATH} \
    --audio_dir ${AUDIO_DIR} \
    --output_path ${EVAL_STYLE}

# Final result summary
echo "=================== Evaluation Completed ==================="
echo "Content Evaluation Results: ${EVAL_CONTENT}"
echo "Style Evaluation Results: ${EVAL_STYLE}"