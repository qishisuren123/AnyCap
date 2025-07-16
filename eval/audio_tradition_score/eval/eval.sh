#!/bin/bash

echo "Running evaluation..."
python audio_tradition_score/eval/eval.py \
    --pred_file "pred/caption/jsonl" \
    --ref_file "gt/caption/jsonl"
echo "Evaluation completed."
