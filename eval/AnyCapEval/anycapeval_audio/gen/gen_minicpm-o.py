# import json
# import torch
# import librosa
# from transformers import AutoModel, AutoTokenizer
# from tqdm import tqdm

# # === 加载模型和tokenizer ===
# model_path = ''
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
#     attn_implementation='sdpa', torch_dtype=torch.bfloat16)
# model = model.eval().cuda()
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model.init_tts()
# model.tts.float()

# # === 文件路径 ===
# input_path = ''
# audio_base_path = ''
# output_path = ''

# # === 读取输入文件并处理每条数据 ===
# with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
#     for line in tqdm(f_in, desc="Processing audio items"):
#         data = json.loads(line)

#         # 提取prompt
#         prompt = None
#         for conv in data['conversations']:
#             if conv['from'] == 'human':
#                 prompt = conv['value']
#                 break
#         if prompt is None:
#             print(f"Skipping id {data['id']} due to missing prompt.")
#             continue

#         # 加载对应的音频
#         audio_path = f"{audio_base_path}/{data['audio']}"
#         try:
#             audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
#         except Exception as e:
#             print(f"Error loading audio for id {data['id']}: {e}")
#             continue

#         # 构造模型输入
#         msgs = [{'role': 'user', 'content': [prompt, audio_input]}]

#         # 调用模型
#         try:
#             response = model.chat(
#                 msgs=msgs,
#                 tokenizer=tokenizer,
#                 sampling=True,
#                 max_new_tokens=128,
#                 use_tts_template=True,
#                 generate_audio=False,  
#                 temperature=0.3,
#             )
#             data['model_response_content'] = response if isinstance(response, str) else getattr(response, 'text', '')
#         except Exception as e:
#             print(f"Error during model chat for id {data['id']}: {e}")
#             continue

#         # 写入更新后的数据
#         f_out.write(json.dumps(data, ensure_ascii=False) + '\n')


import argparse
import json
import torch
import librosa
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import os

def load_minicpm_model(checkpoint, device):
    """Load MiniCPM model for audio processing"""
    model = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "auto" else {"": device}
    )
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    
    # Initialize TTS if needed
    model.init_tts()
    model.tts.float()
    
    return model, tokenizer

def merge_and_sort_outputs(content_file, style_file, output_file):
    """Merge two JSONL files and sort by 'id' in ascending order"""
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

def process_audio_data(args):
    """Process audio data with MiniCPM model"""
    with torch.cuda.device(0):
        model, tokenizer = load_minicpm_model(args.checkpoint, "cuda:0")
    
    content_restriction = ['action', 'appearance', 'background', 'camera', 'event', 'instance', 'perspective', 'position']
    style_restriction = ['brief', 'detail', 'narrative', 'poem', 'theme']

    with open(args.data_path, 'r', encoding='utf-8') as infile, \
         open(args.output_path_content, 'w', encoding='utf-8') as outfile_content, \
         open(args.output_path_style, 'w', encoding='utf-8') as outfile_style:
        
        for line in tqdm(infile, desc="Processing audio items", unit="line"):
            data = json.loads(line.strip())
            
            # Extract prompt and restriction
            prompt = None
            for conv in data['conversations']:
                if conv['from'] == 'human':
                    prompt = conv['value']
                    break
            if prompt is None:
                print(f"Skipping id {data['id']} due to missing prompt.")
                continue
            
            restriction = data['restriction'][0] if 'restriction' in data else 'detail'
            
            # Load audio
            audio_path = f"{args.audio_dir}/{data['audio']}"
            try:
                audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
            except Exception as e:
                print(f"Error loading audio for id {data['id']}: {e}")
                continue
            
            # Process with model
            with torch.no_grad():
                with torch.cuda.device(0):
                    msgs = [{'role': 'user', 'content': [prompt, audio_input]}]
                    
                    try:
                        response = model.chat(
                            msgs=msgs,
                            tokenizer=tokenizer,
                            sampling=args.sample,
                            max_new_tokens=args.max_new_tokens,
                            use_tts_template=True,
                            generate_audio=False,
                            temperature=0.0 if not args.sample else args.temperature,
                        )
                        response_text = response if isinstance(response, str) else getattr(response, 'text', '')
                    except Exception as e:
                        print(f"Error during model chat for id {data['id']}: {e}")
                        continue
            
            # Classify and write output based on restriction
            if restriction in content_restriction:
                data_out = data.copy()
                data_out['model_response_content'] = response_text
                outfile_content.write(json.dumps(data_out, ensure_ascii=False) + '\n')
                outfile_content.flush()
            elif restriction in style_restriction:
                data_out = data.copy()
                data_out['model_response_style'] = response_text
                outfile_style.write(json.dumps(data_out, ensure_ascii=False) + '\n')
                outfile_style.flush()
            else:
                print(f"Restriction '{restriction}' is not in the storage list and cannot be processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniCPM audio processing tool")
    parser.add_argument('--checkpoint', type=str, default='/mnt/petrelfs/renyiming/lzq_workspace/new_bench/checkpoint/MiniCPM-o-2_6',
                       help='MiniCPM model path')
    
    parser.add_argument('--data_path', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/anycapeval_audio_ref.jsonl',
                       help='Path to the JSONL data file')
    parser.add_argument('--audio_dir', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/test_audio_data',
                       help='Directory containing the audio files')
                       
    parser.add_argument('--output_path_content', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/eval/input/content_input/minicpm_content.jsonl',
                       help='Path for content outputs')
    parser.add_argument('--output_path_style', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/eval/input/style_input/minicpm_style.jsonl',
                       help='Path for style outputs')
    parser.add_argument("--merged_output", type=str,
                       default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/eval/input/merged_results.jsonl',
                       help="Path for merged and sorted output file")

    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--sample", action="store_true", help="Enable random sampling")
    parser.add_argument("--auto", action="store_true", help="Enable automatic device mapping")
    
    args = parser.parse_args()

    # Create output directories if they don't exist
    for output_path in [args.output_path_content, args.output_path_style]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("")

    process_audio_data(args)

    # Merge and sort the output files
    merge_and_sort_outputs(args.output_path_content, args.output_path_style, args.merged_output)