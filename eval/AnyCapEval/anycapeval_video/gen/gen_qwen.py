import argparse
import json
import math
import torch
from PIL import Image
import base64
import requests
import torchvision.transforms as T
from tqdm import tqdm
from decord import VideoReader, cpu
import numpy as np
import os

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_qwen_model(checkpoint, device):
    """Load Qwen2.5-VL model"""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto" if device == "auto" else {"": device}
    )
    
    processor = AutoProcessor.from_pretrained(checkpoint)
    
    model.eval()
    return model, processor

def process_data(args):

    with torch.cuda.device(0):
        model, processor = load_qwen_model(args.checkpoint, "cuda:0")
        
    content_restriction = ['action', 'appearance', 'background', 'camera', 'event', 'instance', 'perspective', 'position']
    style_restriction = ['brief', 'detail', 'narrative', 'poem', 'theme']

    with open(args.data_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    with open(args.output_path_content, 'w', encoding='utf-8') as outfile_content, \
         open(args.output_path_style, 'w', encoding='utf-8') as outfile_style:
        for line in tqdm(lines, desc="Processing JSONL", unit="line"):
            data = json.loads(line.strip())
       
            video_path = f"{args.video_dir}/{data['video']}" 
            question = data['conversations'][0]['value']  
            ref_answer = data['conversations'][1]['value']
            restriction = data['restriction'][0] if 'restriction' in data else 'detail'

            with torch.no_grad():
                with torch.cuda.device(0):
                    
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": video_path,
                                    "max_pixels": 640 * 480,
                                    "fps": 8.0,
                                },
                                {"type": "text", "text": f"{question}"},
                            ],
                        }
                    ]
                    
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        fps=8.0,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to("cuda")
                    
                    generated_ids = model.generate(
                        **inputs,
                        do_sample=args.sample,
                        temperature=0.0 if not args.sample else 1.0,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                    )
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0].strip()
                
            if restriction in content_restriction:
                type_restriction = 'content'
                data_out = data.copy()
                data_out['model_response_content'] = response
                outfile_content.write(json.dumps(data_out, ensure_ascii=False) + '\n')
                outfile_content.flush()

            elif restriction in style_restriction:
                type_restriction = 'style'
                data_out = data.copy()
                data_out['model_response_style'] = response
                outfile_style.write(json.dumps(data_out, ensure_ascii=False) + '\n')
                outfile_style.flush()            
            else:
                print('Restriction is not in the storage list and cannot be processed.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-VL video processing tool")
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct',
                       help='Qwen2.5-VL model path')

    parser.add_argument('--data_path', type=str, 
                       default='/mnt/petrelfs/renyiming/ly_workspace/submit_code/anycapeval_video/anycapeval_video_ref.jsonl',
                       help='Path to the JSONL data file')
    parser.add_argument('--video_dir', type=str, 
                       default='/mnt/petrelfs/renyiming/ly_workspace/submit_code/anycapeval_video/video_datas',
                       help='Directory containing the videos')
                       
    parser.add_argument('--output_path_content', type=str, default='/mnt/petrelfs/renyiming/ly_workspace/submit_code/anycapeval_video/output/tmp_qwen_content',
                       help='Path for content outputs')
    parser.add_argument('--output_path_style', type=str, default='/mnt/petrelfs/renyiming/ly_workspace/submit_code/anycapeval_video/output/tmp_qwen_style',
                       help='Path for style outputs')

    parser.add_argument("--num-beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--sample", action="store_true", help="Enable random sampling")
    parser.add_argument("--auto", action="store_true", help="Enable automatic device mapping")
    args = parser.parse_args()

    for output_path in [args.output_path_content, args.output_path_style]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("")

    process_data(args)