# import base64
# import json
# import os
# import random
# import wave
# from openai import OpenAI

# # 初始化OpenAI客户端
# client = OpenAI(
#     api_key=''
# )

# # 文件路径配置
# BASE_INPUT_PATH = ""
# BASE_OUTPUT_PATH = ""

# def get_data_in_id_range(file_path, id_range=(0, 101)):
#     """获取指定ID范围内的数据"""
#     data_list = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             data = json.loads(line)
#             if id_range[0] <= data['id'] <= id_range[1]:
#                 data_list.append(data)
#     return data_list

# def process_audio_and_get_gpt_response(audio_path, question):
#     """使用直连API处理音频并获取响应"""
#     # 读取并编码音频文件
#     with open(audio_path, "rb") as audio_file:
#         encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
    
#     try:
#         completion = client.chat.completions.create(
#             model="gpt-4o-audio-preview",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": question
#                         },
#                         {
#                             "type": "input_audio",
#                             "input_audio": {
#                                 "data": encoded_audio,
#                                 "format": "wav"
#                             }
#                         }
#                     ]
#                 }
#             ]
#         )
#         print (completion.choices[0].message)
#         return completion.choices[0].message.content
#     except Exception as e:
#         print(f"API请求失败: {str(e)}")
#         return None

# def main(input_path, output_path):
#     """主处理函数"""
#     audio_base_dir = '/mnt/petrelfs/renyiming/gm_audio_data/newbench/'

#     # 遍历三种类型处理
#     data_list = get_data_in_id_range(input_path, id_range=(0, 101))
    
#     with open(output_path, 'w') as output_file:
#         for entry in data_list:
#             audio_key = entry["audio"]
#             audio_path = os.path.join(audio_base_dir, audio_key)
#             time_long = entry["time_long"]
#             # 直接从数据中获取问题和回答
#             conversations = entry['conversations']
#             question = conversations[0]['value']
#             ref_response = conversations[1]['value']
            
#             response = process_audio_and_get_gpt_response(audio_path, question)
#             if response:
#                 output_entry = {
#                     "id": entry["id"],
#                     "audio": os.path.basename(audio_path),
#                     "conversations": [
#                         {"from": "human", "value": question},
#                         {"from": "gpt", "value": ref_response}
#                     ],
#                     "time_long": time_long,
#                     "restriction": entry["restriction"],
#                     "model_response_content": response
#                 }
#                 output_file.write(json.dumps(output_entry) + '\n')
#             else:
#                 print("处理失败，跳过该条目")

# if __name__ == "__main__":
#     main(BASE_INPUT_PATH, BASE_OUTPUT_PATH)


import argparse
import json
import os
import base64
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Any

def parse_arguments():
    """Parse command line arguments (与图片模态对齐)"""
    parser = argparse.ArgumentParser(description="GPT-4o模型生成音频caption(语音模态)")
    
    # API configuration parameters
    parser.add_argument('--api_key', type=str, 
                        default='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',
                        help='OpenAI API key')
    parser.add_argument('--model', type=str, default="gpt-4o-audio-preview",
                        help='Model name to use')
    
    # Data path parameters
    parser.add_argument('--data_path', type=str, 
                       default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/test_ref.jsonl',
                       help='Path to the JSONL data file')
    parser.add_argument('--audio_dir', type=str, 
                       default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/test_audio_data',
                       help='Directory containing the audio files')
    parser.add_argument('--output_path_content', type=str, 
                       default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/eval/output/temp_content.jsonl',
                       help='Path for content-related outputs')
    parser.add_argument('--output_path_style', type=str, 
                       default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/eval/output/temp_style.jsonl',
                       help='Path for non-content-related outputs')
    parser.add_argument("--merged_output", type=str,
                       default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/eval/output/merged_results.jsonl',
                       help="Path for merged and sorted output file")
    
    # Other parameters
    parser.add_argument('--max_tokens', type=int, default=2000,
                        help='Maximum tokens for API response')
    parser.add_argument('--max_retries', type=int, default=5,
                        help='Maximum retries for failed API calls')
    parser.add_argument('--id_range', type=str, default="0-100",
                        help='ID range to process (e.g. "0-100")')
    
    return parser.parse_args()

def is_id_exists(file_path: str, id: int) -> bool:
    if not os.path.exists(file_path):
        return False
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            if data.get('id') == id:
                return True
    return False

def process_audio_and_get_gpt_response(audio_path: str, question: str, model: str, max_retries: int, api_key: str) -> str:
    """使用直连API处理音频并获取响应 (增加重试机制)"""
    client = OpenAI(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            # 读取并编码音频文件
            with open(audio_path, "rb") as audio_file:
                encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
            
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": encoded_audio,
                                    "format": "wav"
                                }
                            }
                        ]
                    }
                ]
            )
            return completion.choices[0].message.content
        
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"API请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
    
    return None

def process_data(args):
    content_restriction = ['event']
    style_restriction = ['brief', 'narrative', 'poem']
    
    # 解析ID范围
    id_start, id_end = map(int, args.id_range.split('-'))
    
    with open(args.data_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    with open(args.output_path_content, 'w', encoding='utf-8') as outfile_content, \
         open(args.output_path_style, 'w', encoding='utf-8') as outfile_style:
        
        for line in tqdm(lines, desc=f"Processing audios with {args.model}", unit="audio"):
            data = json.loads(line.strip())
            audio_path = f"{args.audio_dir}/{data['audio']}"
            question = data['conversations'][0]['value']
            restriction = data['restriction'][0] if 'restriction' in data else 'detail'
            id = data.get('id')
            
            # 检查ID是否在范围内
            if not (id_start <= id <= id_end):
                continue
                
            # 检查id是否已经存在
            if (os.path.exists(args.output_path_content) and is_id_exists(args.output_path_content, id)) or \
               (os.path.exists(args.output_path_style) and is_id_exists(args.output_path_style, id)):
                print(f"ID {id} already exists. Skipping...")
                continue
            
            try:
                response = process_audio_and_get_gpt_response(
                    audio_path, question, args.model, args.max_retries, args.api_key
                )
                print(response)
                result = data.copy()
                
                if restriction in content_restriction:
                    result['model_response_content'] = response
                    outfile_content.write(json.dumps(result, ensure_ascii=False) + '\n')
                    outfile_content.flush()
                elif restriction in style_restriction:
                    result['model_response_style'] = response
                    outfile_style.write(json.dumps(result, ensure_ascii=False) + '\n')
                    outfile_style.flush()
                else:
                    print(f'Restriction type "{restriction}" not in storage list, cannot process.')
                    
            except Exception as e:
                print(f"Error processing audio {data['audio']}: {str(e)}")
                # Log error but continue processing
                error_result = data.copy()
                error_result['error'] = str(e)
                if restriction in content_restriction:
                    error_result['model_response_content'] = ["Error: " + str(e)]
                    outfile_content.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                    outfile_content.flush()
                elif restriction in style_restriction:
                    error_result['model_response_style'] = ["Error: " + str(e)]
                    outfile_style.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                    outfile_style.flush()

def merge_and_sort_outputs(content_file: str, style_file: str, output_file: str):
    all_data = []
    
    # Read content file
    if os.path.exists(content_file):
        with open(content_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    all_data.append(data)
    
    # Read style file
    if os.path.exists(style_file):
        with open(style_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    all_data.append(data)
    
    # Sort by 'id' in ascending order
    all_data.sort(key=lambda x: x.get('id', 0))
    
    # Write merged and sorted data
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in all_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Merged {len(all_data)} records and saved to {output_file}")

def main():
    args = parse_arguments()
    
    for output_path in [args.output_path_content, args.output_path_style, args.merged_output]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    print(f"Starting audio analysis with {args.model}...")
    process_data(args)
    print(f"Processing completed. Content outputs saved to: {args.output_path_content}")
    print(f"Style outputs saved to: {args.output_path_style}")
    
    merge_and_sort_outputs(args.output_path_content, args.output_path_style, args.merged_output)

if __name__ == "__main__":
    main()