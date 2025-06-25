# import os
# import json
# import argparse
# import torch
# import torchaudio
# from transformers import AutoTokenizer
# from internvl.model.internvl_chat import InternVLChatModel

# def get_audio_duration(audio_path):
#     try:
#         waveform, sample_rate = torchaudio.load(audio_path)
#         return waveform.shape[1] / sample_rate
#     except Exception as e:
#         print(f"Error loading audio {audio_path}: {str(e)}")
#         return None

# def test_internvl_chat_audio_integration(input_file, output_jsonl, audio_base_dir, checkpoint):

#     max_new_tokens = 100
#     temperature = 0.0
#     load_in_8bit = False
#     load_in_4bit = False
#     auto_device_mapping = True

#     torch.cuda.set_device(0)
#     device = torch.device("cuda:0")

#     kwargs = {'device_map': 'auto'} if auto_device_mapping else {}
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
#     model = InternVLChatModel.from_pretrained(
#         checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
#         load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs).eval()

#     if not load_in_8bit and not load_in_4bit and not auto_device_mapping:
#         model = model.cuda()

#     model._load_audio_model()

#     os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
#     with open(output_jsonl, 'w', encoding='utf-8') as f_out:
#         try:
#             with open(input_file, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     base_data = json.loads(line)
#                     audio_key = base_data['audio']
#                     audio_path = os.path.join(audio_base_dir, audio_key)
                    
#                     question = base_data['conversations'][0]['value']
#                     response = base_data['conversations'][1]['value']
                    
#                     prompt = (f"<audio>\nYou are a multimodal aligner. Transform the existing caption to meet "
#                           f"the requirement: '{question[8:]}'. Existing caption: '{response}'. "
#                           f"Only respond with the improved caption.")

#                     try:
#                         audio_tensor = model.audio_model.load_audio(audio_path).cuda()
#                         time_long = get_audio_duration(audio_path)
#                         if time_long is None:
#                             continue
#                     except Exception as e:
#                         print(f"Error loading audio {audio_path}: {str(e)}")
#                         continue

#                     try:
#                         generation_config = dict(
#                             num_beams=1,
#                             max_new_tokens=max_new_tokens,
#                             min_new_tokens=1,
#                             do_sample=True if temperature > 0 else False,
#                             temperature=temperature,
#                         )

#                         pred = model.chat(
#                             tokenizer=tokenizer,
#                             pixel_values=None,
#                             audio_values=audio_tensor,
#                             question=prompt,
#                             generation_config=generation_config
#                         )
                        
#                         result = {
#                             "id": base_data['id'],
#                             "audio": audio_key,
#                             "conversations": [
#                                 {"from": "human", "value": question},
#                                 {"from": "gpt", "value": response}
#                             ],
#                             "time_long": time_long,
#                             "restriction": base_data['restriction'],
#                             "model_response_content": pred
#                         }
                        
#                         f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
#                     except Exception as e:
#                         print(f"Processing error: {str(e)}")
#         except FileNotFoundError as e:
#             print(f"File not found: {input_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
#     parser.add_argument("--output_file", type=str, required=True, help="Path to save output JSONL")
#     parser.add_argument("--audio_dir", type=str, required=True, help="Base directory for audio files")
#     parser.add_argument("--checkpoint", type=str, required=True, help="Path to InternVL checkpoint")

#     args = parser.parse_args()

#     test_internvl_chat_audio_integration(
#         args.input_file,
#         args.output_file,
#         args.audio_dir,
#         args.checkpoint
#     )



import os
import json
import argparse
import torch
import torchaudio
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from internvl.model import InternVLChatModel
from decord import VideoReader, cpu
import numpy as np


# content/style分类
content_restriction = ['action', 'appearance', 'background', 'camera', 'event', 'instance', 'perspective', 'position']
style_restriction = ['brief', 'detail', 'narrative', 'poem', 'theme']

def get_most_available_gpu():
    gpu_memory = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        gpu_memory.append((i, torch.cuda.memory_allocated(i)))
    gpu_memory.sort(key=lambda x: x[1])
    return f"cuda:{gpu_memory[0][0]}"

def load_responses_from_jsonl(file_path):
    responses = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():  
                    continue
                try:
                    data = json.loads(line.strip())
                    video_id = data['id']
                    if 'model_response' in data:
                        responses[video_id] = data['model_response']
                    elif 'model_response_content' in data:
                        responses[video_id] = data['model_response_content']
                    elif 'model_response_style' in data:
                        responses[video_id] = data['model_response_style']
                    elif 'conversations' in data and len(data['conversations']) > 1:
                        responses[video_id] = data['conversations'][1]['value']
                except json.JSONDecodeError:
                    print(f"Error parsing JSON line: {line}")
                    continue
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
    return responses

def load_internvl_model(checkpoint, device, args):
    if args.auto:
        config = InternVLChatConfig.from_pretrained(checkpoint)
        device_map = split_model(config.llm_config.num_hidden_layers)
    else:
        device_map = {"": device}

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, 
        trust_remote_code=True,
        use_fast=False
    )

    model = InternVLChatModel.from_pretrained(
        checkpoint,
        device_map=device_map if args.auto else None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    ).eval()
    
    if not args.auto and not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(device)
    
    # 加载音频模型
    model._load_audio_model()
    
    return model, tokenizer

def get_audio_duration(audio_path):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform.shape[1] / sample_rate
    except Exception as e:
        print(f"Error loading audio {audio_path}: {str(e)}")
        return None

def process_audio_data(args):
    device = get_most_available_gpu()
    model, tokenizer = load_internvl_model(args.checkpoint, device, args)
    device = model.device

    pregenerated_responses = load_responses_from_jsonl(args.init_responses_path)
    if not pregenerated_responses:
        raise ValueError(f"Failed to load any responses from {args.init_responses_path}")
    print(f"Successfully loaded {len(pregenerated_responses)} pre-generated responses")

    with open(args.data_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    with open(args.output_path_style, 'w', encoding='utf-8') as outfile_style, \
         open(args.output_path_content, 'w', encoding='utf-8') as outfile_content:
        
        for line in tqdm(lines, desc="Processing JSONL", unit="line"):
            if not line.strip():  
                continue
                
            data = json.loads(line.strip())
            audio_path = f"{args.audio_dir}/{data['audio']}" 
            audio_id = data['id']
            question = data['conversations'][0]['value']  
            restriction = data['restriction'][0] if 'restriction' in data else 'detail'

            if audio_id not in pregenerated_responses:
                print(f"Warning: Audio ID {audio_id} not found in pre-generated responses, skipping")
                continue
            
            init_response = pregenerated_responses[audio_id]

            try:
                audio_tensor = model.audio_model.load_audio(audio_path).to(device)
                duration = get_audio_duration(audio_path)
                if duration is None:
                    continue
            except Exception as e:
                print(f"Error loading audio {audio_path}: {str(e)}")
                continue

            gen_config = {
                "do_sample": args.sample,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "num_beams": args.num_beams,
                "eos_token_id": tokenizer.eos_token_id
            }
            
            prompt = f"<audio>\nYou are a multimodal aligner. Transform the existing caption to meet " \
                    f"the requirement: '{question}'. Existing caption: '{init_response}'. " \
                    "Only respond with the improved caption."

            with torch.no_grad():
                response = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=None,
                    audio_values=audio_tensor,
                    question=prompt,
                    generation_config=gen_config
                )
            
            if restriction in content_restriction:
                output_data = {
                    "id": audio_id,
                    "audio": data['audio'],
                    "conversations": data['conversations'],
                    "time_long": duration,
                    "restriction": data['restriction'],
                    "model_response_content": response
                }
                outfile_content.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                outfile_content.flush()
            elif restriction in style_restriction:
                output_data = {
                    "id": audio_id,
                    "audio": data['audio'],
                    "conversations": data['conversations'],
                    "time_long": duration,
                    "restriction": data['restriction'],
                    "model_response_style": response
                }
                outfile_style.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                outfile_style.flush()
            else:
                print(f'Restriction "{restriction}" not recognized, skipping.')

def merge_and_sort_outputs(content_file, style_file, output_file):
    all_data = []
    
    if os.path.exists(content_file):
        with open(content_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    all_data.append(data)
    
    if os.path.exists(style_file):
        with open(style_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    all_data.append(data)
    
    all_data.sort(key=lambda x: x.get('id', 0))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in all_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Merged {len(all_data)} records to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 必须参数
    parser.add_argument('--checkpoint', type=str, default='/mnt/petrelfs/renyiming/model/InternVL2/internvl_chat_audio/work_dirs/internvl_chat_v3_0/internvl3_2b_221_221_221',
                       help='InternVL模型checkpoint路径')
    parser.add_argument('--init_responses_path', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/eval/input/merged_results.jsonl',
                       help='预生成响应的JSONL文件路径')
    parser.add_argument('--data_path', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/anycapeval_audio_ref.jsonl',
                       help='输入数据JSONL文件路径')
    parser.add_argument('--audio_dir', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/test_audio_data',
                       help='音频文件目录')
    
    # 输出参数
    parser.add_argument('--output_path_content', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/eval/input/content_input/acm_content.jsonl',
                       help='内容类输出JSONL路径')
    parser.add_argument('--output_path_style', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/eval/input/style_input/acm_style.jsonl',
                       help='风格类输出JSONL路径')
    parser.add_argument('--merged-output', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/eval/input/acm_merged_results.jsonl',
                       help='合并后的输出文件路径')
    
    # 生成参数
    parser.add_argument("--num-beams", type=int, default=1, help="beam search的beam数")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k采样参数")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p采样参数")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="生成的最大token数")
    parser.add_argument("--sample", action="store_true", help="启用随机采样")
    
    # 模型加载参数
    parser.add_argument("--load-in-8bit", action="store_true", help="启用8-bit量化")
    parser.add_argument("--load-in-4bit", action="store_true", help="启用4-bit量化")
    parser.add_argument("--auto", action="store_true", help="启用自动设备映射")

    args = parser.parse_args()

    # 确保输出目录存在（与视频模态相同）
    for output_path in [args.output_path_content, args.output_path_style]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("")

    # 执行处理流程
    process_audio_data(args)
    merge_and_sort_outputs(args.output_path_content, args.output_path_style, args.merged_output)