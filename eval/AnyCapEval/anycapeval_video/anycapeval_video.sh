#!/bin/bash

# Set base paths
BASE_DIR="/mnt/petrelfs/renyiming/ly_workspace/submit_code/anycapeval_video"
SCRIPT_DIR="/mnt/petrelfs/renyiming/ly_workspace/submit_code/anycapeval_video/eval"  # Please replace with actual script directory
OUTPUT_DIR="${BASE_DIR}/output"

EXP_ID="code_test"  
OUTPUT_FOLDER="${OUTPUT_DIR}/${EXP_ID}/"

# Create output directory
mkdir -p ${OUTPUT_FOLDER}

# Set output paths
OUTPUT_PATH_CONTENT="/mnt/petrelfs/renyiming/ly_workspace/submit_code/anycapeval_video/output/temp_content.jsonl"
OUTPUT_PATH_STYLE="/mnt/petrelfs/renyiming/ly_workspace/submit_code/anycapeval_video/output/temp_style.jsonl"

# Set evaluation result paths
EVAL_CONTENT="${OUTPUT_FOLDER}/eval_content.jsonl"
EVAL_STYLE="${OUTPUT_FOLDER}/eval_style.jsonl"

# API configuration
API_BASE_URL="https://boyuerichdata.chatgptten.com/v1/"
API_KEY="sk-iproK7tAwu7J2ZBJWL8G3TiKUepPUH6uj5JQ7w0oXCRu02wl"
EVAL_CONTENT_MODEL="gpt-4o-mini"
EVAL_STYLE_MODEL="gpt-4o-2024-08-06"

# Other fixed paths
VIDEO_DIR="${BASE_DIR}/video_datas"
# DATA_PATH="${BASE_DIR}/anycapeval_video_ref.jsonl"
CONTENT_DATA_PATH="${BASE_DIR}/anycapeval_video_content.jsonl"
CONTENT_TEMPLATE_PATH="${BASE_DIR}/instruction_content.txt"
STYLE_TEMPLATE_PATH="${BASE_DIR}/instruction_style.txt"

echo "=================== Starting Evaluation Process ==================="
echo "Experiment ID: ${EXP_ID}"
echo "Output Directory: ${OUTPUT_FOLDER}"


# Check if generation was successful
if [ ! -f "${OUTPUT_PATH_CONTENT}" ] || [ ! -f "${OUTPUT_PATH_STYLE}" ]; then
    echo "Error: Response file generation failed, please check logs"
    exit 1
fi

# echo "=================== Step 1: Evaluating Content Quality ==================="
# python ${SCRIPT_DIR}/anycapeval_video_content.py \
#     --base_url ${API_BASE_URL} \
#     --api_key ${API_KEY} \
#     --model ${EVAL_CONTENT_MODEL} \
#     --data_path ${OUTPUT_PATH_CONTENT} \
#     --content_data_path ${CONTENT_DATA_PATH} \
#     --content_template_path ${CONTENT_TEMPLATE_PATH} \
#     --video_dir ${VIDEO_DIR} \
#     --output_path ${EVAL_CONTENT}

echo "=================== Step 2: Evaluating Style ==================="
python ${SCRIPT_DIR}/anycapeval_video_style.py \
    --base_url ${API_BASE_URL} \
    --api_key ${API_KEY} \
    --model ${EVAL_STYLE_MODEL} \
    --data_path ${OUTPUT_PATH_STYLE} \
    --content_template_path ${STYLE_TEMPLATE_PATH} \
    --video_dir ${VIDEO_DIR} \
    --output_path ${EVAL_STYLE}

# Final result summary
echo "=================== Evaluation Completed ==================="
echo "Content Evaluation Results: ${EVAL_CONTENT}"
echo "Style Evaluation Results: ${EVAL_STYLE}"