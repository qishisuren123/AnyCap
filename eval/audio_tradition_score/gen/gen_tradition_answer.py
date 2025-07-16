import os
import json
import torch
import argparse
from transformers import AutoTokenizer
from internvl.model.internvl_chat import InternVLChatModel

def test_internvl_chat_audio_integration(args):
    # 设置GPU设备
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    # 初始化tokenizer和模型
    kwargs = {'device_map': 'auto'} if args.auto_device_mapping else {}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, **kwargs).eval()

    # 调整模型设备映射
    if not args.load_in_8bit and not args.load_in_4bit and not args.auto_device_mapping:
        model = model.cuda()

    # 添加音频token
    AUDIO_START_TOKEN = '<audio>'
    AUDIO_END_TOKEN = '</audio>'
    AUDIO_CONTEXT_TOKEN = '<AUDIO_CONTEXT>'
    token_list = [AUDIO_START_TOKEN, AUDIO_END_TOKEN, AUDIO_CONTEXT_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # 加载音频模型
    model._load_audio_model()

    # 读取输入JSONL文件
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 创建输出目录（如果需要）
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 处理并保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for line in lines:
            data = json.loads(line)
            audio_path = data['audio_path'] #os.path.join(args.audio_dir, data['audio'])
            question = "<audio>\nCan you provide a brief caption about this audio?"
            
            print(f"Processing file: {audio_path}")
            
            try:
                audio_tensor = model.audio_model.load_audio(audio_path).cuda()

                generation_config = dict(
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=1,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                )

                pred = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=None,
                    audio_values=audio_tensor,
                    question=question,
                    generation_config=generation_config
                )

                result = {
                    'audio_path': audio_path,
                    'model_response': pred
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                print(f"Saved prediction for {audio_path}")

            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--input_file', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Max new tokens')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--load_in_8bit', action='store_true', help='Load model in 8bit')
    parser.add_argument('--load_in_4bit', action='store_true', help='Load model in 4bit')
    parser.add_argument('--auto_device_mapping', action='store_true', help='Use auto device mapping')
    args = parser.parse_args()

    test_internvl_chat_audio_integration(args)
