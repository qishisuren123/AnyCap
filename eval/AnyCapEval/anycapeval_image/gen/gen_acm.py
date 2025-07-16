import argparse
import json
import math
import os
import torch
from PIL import Image
from internvl.train.dataset import build_transform, dynamic_preprocess
from internvl.model import InternVLChatConfig, InternVLChatModel
from transformers import AutoTokenizer
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def is_id_exists(file_path, id):
    if not os.path.exists(file_path):
        return False
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                if data.get('id') == id:
                    return True
            except json.JSONDecodeError:
                continue
    return False

def load_image(image_path, input_size, device, dynamic=False, max_num=6):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(is_train=False, input_size=input_size)
    
    if dynamic:
        processed_images = dynamic_preprocess(
            image, 
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num
        )
    else:
        processed_images = [image]
    
    pixel_values = torch.stack([transform(img) for img in processed_images])
    return pixel_values.to(device)

def split_model(num_layers, vit_alpha=0.5):
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

def load_responses_from_jsonl(file_path):
    responses = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():  
                    continue
                try:
                    data = json.loads(line.strip())
                    if 'id' not in data:
                        continue
                    if 'model_response' in data:
                        responses[data['id']] = data['model_response']
                    elif 'model_response_content' in data:
                        responses[data['id']] = data['model_response_content']
                    elif 'model_response_style' in data:
                        responses[data['id']] = data['model_response_style']
                    elif 'conversations' in data and len(data['conversations']) > 1:
                        responses[data['id']] = data['conversations'][1]['value']
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
    return responses

def get_most_available_gpu():
    gpu_memory = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        gpu_memory.append((i, torch.cuda.memory_allocated(i)))
    gpu_memory.sort(key=lambda x: x[1])
    return f"cuda:{gpu_memory[0][0]}"

def merge_and_sort_outputs(content_file, style_file, output_file):
    all_data = []
    for file_path in [content_file, style_file]:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_data.append(json.loads(line.strip()))
    all_data.sort(key=lambda x: x.get('id', 0))
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in all_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"Merged {len(all_data)} records to {output_file}")

def process_data(args):
    device = get_most_available_gpu()
    model, tokenizer = load_internvl_model(args.checkpoint, device, args)
    
    config = InternVLChatConfig.from_pretrained(args.checkpoint)
    image_size = config.force_image_size or config.vision_config.image_size
    
    content_restriction = ['perspective', 'multi_class_position', 
                          'multi_class_appearance', 'instance']
    style_restriction = ['brief', 'detail', 'narrative', 'poem', 'theme']
    
    for path in [args.output_path_content, args.output_path_style]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(args.data_path, 'r', encoding='utf-8') as infile, \
         open(args.output_path_content, 'a', encoding='utf-8') as out_content, \
         open(args.output_path_style, 'a', encoding='utf-8') as out_style:
        
        for line in tqdm(infile, desc="Processing Images"):
            if not line.strip():
                continue
                
            data = json.loads(line.strip())
            image_path = os.path.join(args.image_dir, data['image'])
            id = data['id']
            
            output_path = args.output_path_content if data.get('restriction', [''])[0] in content_restriction else args.output_path_style
            if is_id_exists(output_path, id):
                continue
            
            pixel_values = load_image(
                image_path, 
                input_size=image_size,
                device=device,
                dynamic=args.dynamic,
                max_num=args.max_num
            ).to(torch.bfloat16)
            
            gen_config = {
                "do_sample": args.sample,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "num_beams": args.num_beams,
                "eos_token_id": tokenizer.eos_token_id
            }
            
            init_response = data.get('model_response', '')
            question = data['conversations'][0]['value']
            
            with torch.no_grad():
                prompt = (f"<image>\nYou are a multimodal aligner. Transform the existing caption to meet "
                          f"the requirement: '{question}'. Existing caption: '{init_response}'. "
                          f"Only respond with the improved caption.")
                response = model.chat(tokenizer, pixel_values, prompt, gen_config)
            
            output_data = data.copy()
            if data.get('restriction', [''])[0] in content_restriction:
                output_data['model_response_content'] = response
                out_content.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                out_content.flush()
            else:
                output_data['model_response_style'] = response
                out_style.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                out_style.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, default='path/to/ACM/model/checkpoint', help='Path to model checkpoint file')
    parser.add_argument('--data_path', type=str, default='path/to/base/model/output.jsonl', help='Path to the JSONL file containing pregenerated responses')
    parser.add_argument('--image_dir', type=str, default='path/to/test/image/directory', help='Directory containing image files')
    parser.add_argument('--output-path-content', type=str, default='path/to/output/content.jsonl', help='Path for content-related outputs JSONL')
    parser.add_argument('--output-path-style', type=str, default='path/to/output/style.jsonl', help='Path for style-related outputs JSONL')
    parser.add_argument('--merged-output', type=str, default='path/to/output/merged_results.jsonl', help='Path for merged output JSONL file')

    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--max-num", type=int, default=6)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--auto", action="store_true")
    
    args = parser.parse_args()
    
    process_data(args)
    
    if args.merged_output:
        merge_and_sort_outputs(
            args.output_path_content,
            args.output_path_style,
            args.merged_output
        )