#!/bin/bash

echo "Running evaluation..."
python evaluate.py \
    --pred_file "这里是模型生成caption文件的地址" \
    --ref_file "这里是标答caption文件的地址"
echo "Evaluation completed."
