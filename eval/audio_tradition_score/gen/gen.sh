#!/bin/bash

python audio_tradition_score/gen/gen_tradition_answer.py \
    --checkpoint "" \
    --input_file "/mnt/petrelfs/renyiming/gm_workspace/gm_workspace_P/clotho_eval/clotho_captions_evaluation_1_try.jsonl" \
    --output_file "/mnt/petrelfs/renyiming/gm_workspace/gm_workspace_P/clotho_eval/predictions_internvl3_2b_221_221_221_test.jsonl" \
    --audio_dir "/mnt/petrelfs/renyiming/audio_data" \
    --max_new_tokens 100 \
    --temperature 0.0 \
    --auto_device_mapping