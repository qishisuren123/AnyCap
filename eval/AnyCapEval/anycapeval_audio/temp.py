import json
import os

def renumber_ids_inplace(file_path):
    # 1. 先读取所有数据到内存
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f]
    
    # 2. 重新编号
    for new_id, data in enumerate(lines):
        data['id'] = new_id
    
    # 3. 覆盖写入原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        for data in lines:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

# 使用示例（直接覆盖原文件）
input_file = '/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/anycapeval_audio_ref.jsonl'  # 替换为你的文件路径
renumber_ids_inplace(input_file)
print(f"ID 重新编号完成，已覆盖 {input_file}")