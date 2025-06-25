import argparse
import json
import math
import torch
from PIL import Image
from internvl.train.dataset import build_transform, dynamic_preprocess
from internvl.model import InternVLChatConfig, InternVLChatModel
from transformers import AutoTokenizer
from tqdm import tqdm
from decord import VideoReader, cpu
import numpy as np
import os

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamic image preprocessing function"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=16):
    """Calculate video frame indices"""
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=16):
    """Load video and process it into model input format"""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(is_train=False, input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def split_model(num_layers, vit_alpha=0.5):
    """Device mapping generation function"""
    world_size = torch.cuda.device_count()
    effective_gpus = world_size - vit_alpha
    base_layers = math.ceil(num_layers / effective_gpus)
    
    layers_per_gpu = [base_layers] * world_size
    layers_per_gpu[0] = math.ceil(base_layers * (1 - vit_alpha))
    
    device_map = {}
    layer_cnt = 0
    for gpu_idx, num_layers in enumerate(layers_per_gpu):
        for _ in range(num_layers):
            if layer_cnt >= num_layers:
                break
            device_map[f'language_model.model.layers.{layer_cnt}'] = gpu_idx
            layer_cnt += 1
    
    visual_components = [
        'vision_model', 'mlp1', 'language_model.model.tok_embeddings',
        'language_model.model.embed_tokens', 'language_model.output',
        'language_model.model.norm', 'language_model.lm_head',
        f'language_model.model.layers.{num_layers - 1}',
        'language_model.model.rotary_emb'
    ]
    for comp in visual_components:
        device_map[comp] = 0
    
    return device_map

def load_internvl_model(checkpoint, device, args):
    """Load InternVL model"""
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

def process_data(args):

    model, tokenizer = load_internvl_model(args.checkpoint, "cuda:0", args)
    device = model.device

    config = InternVLChatConfig.from_pretrained(args.checkpoint)
    image_size = config.force_image_size or config.vision_config.image_size
    
    content_restriction = ['action', 'appearance', 'background', 'camera', 'event', 'instance', 'perspective', 'position']
    style_restriction = ['brief', 'detail', 'narrative', 'poem', 'theme']

    with open(args.data_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    with open(args.output_path_content, 'w', encoding='utf-8') as outfile_content, \
         open(args.output_path_style, 'w', encoding='utf-8') as outfile_style:
        for line in tqdm(lines, desc="Processing JSONL", unit="line"):
            if not line.strip():  
                continue
                
            data = json.loads(line.strip())
       
            video_path = f"{args.video_dir}/{data['video']}"  
            video_id = data['video']
            id = data['id']
            question = data['conversations'][0]['value']  
            restriction = data['restriction'][0] if 'restriction' in data else 'detail'

            try:
                pixel_values, num_patches_list = load_video(
                    video_path,
                    num_segments=args.num_segments,
                    input_size=image_size,
                    max_num=args.max_num
                )
                
                pixel_values = pixel_values.to(device).to(torch.bfloat16)

                video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))])
                
                gen_config = {
                    "do_sample": args.sample,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "num_beams": args.num_beams,
                    "eos_token_id": tokenizer.eos_token_id
                }
                
                with torch.no_grad():
                    question = f"{video_prefix}{question}"
                    question = [question]
                    response = model.batch_chat_video(
                        tokenizer=tokenizer,
                        pixel_values=pixel_values,
                        questions=question,
                        generation_config=gen_config,
                        num_patches_list=[num_patches_list]
                    )
                
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
                    print(f'Restriction "{restriction}" is not in the storage list and cannot be processed.')
                    
            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Variable parameters
    parser.add_argument('--checkpoint', type=str, 
                       default='/mnt/petrelfs/share_data/wangweiyun/share_internvl/InternVL2_5-8B/',
                       help='Model checkpoint path')
    parser.add_argument('--output_path_content', type=str, 
                       default='/mnt/petrelfs/renyiming/ly_workspace/submit_code/anycapeval_video/output/temp_content.jsonl',
                       help='Path for model content outputs')
    parser.add_argument('--output_path_style', type=str, 
                       default='/mnt/petrelfs/renyiming/ly_workspace/submit_code/anycapeval_video/output/temp_style.jsonl',
                       help='Path for model non-content outputs')
    parser.add_argument("--merged-output", type=str,
                       default='/mnt/petrelfs/renyiming/ly_workspace/submit_code/anycapeval_video/output/merged_results.jsonl',
                       help="Path for merged and sorted output file")
                       
    # Fixed parameters  
    parser.add_argument('--data_path', type=str, 
                       default='/mnt/petrelfs/renyiming/ly_workspace/submit_code/anycapeval_video/anycapeval_video_ref.jsonl',
                       help='Path to the JSONL data file')
    parser.add_argument('--video_dir', type=str, 
                       default='/mnt/petrelfs/renyiming/ly_workspace/submit_code/anycapeval_video/video_datas',
                       help='Directory containing the videos')

    parser.add_argument("--num-beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--sample", action="store_true", help="Enable random sampling")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic image processing")
    parser.add_argument("--max-num", type=int, default=6, help="Maximum number of image patches")
    parser.add_argument("--num-segments", type=int, default=8, help="Number of video frame samples") 
    parser.add_argument("--load-in-8bit", action="store_true", help="Enable 8-bit quantization")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantization")
    parser.add_argument("--auto", action="store_true", help="Enable automatic device mapping")
    args = parser.parse_args()


    for output_path in [args.output_path_content, args.output_path_style]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("")

    process_data(args)
    
    # Merge and sort the output files
    merge_and_sort_outputs(args.output_path_content, args.output_path_style, args.merged_output)