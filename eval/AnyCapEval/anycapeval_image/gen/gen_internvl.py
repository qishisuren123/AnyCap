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

def is_id_exists(file_path, id):
    """Check if the specified ID already exists in the file"""
    if not os.path.exists(file_path):
        return False
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            if data['id'] == id:
                return True
    return False

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
    
    # print(f"Merged {len(all_data)} records and saved to {output_file}")
    print("Merged {} records and saved to {}".format(len(all_data), output_file))

def split_model(num_layers, vit_alpha=0.5):
    """Device mapping generation function (aligned with video version)"""
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
    """Model loading function (aligned with video version)"""
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

def load_image(image_path, input_size, device, args):
    """Image loading function (aligned with video version)"""
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(is_train=False, input_size=input_size)
    
    if args.dynamic:
        processed_images = dynamic_preprocess(
            image, 
            image_size=input_size,
            use_thumbnail=True,
            max_num=args.max_num
        )
    else:
        processed_images = [image]
    
    pixel_values = torch.stack([transform(img) for img in processed_images])
    return pixel_values.to(device).to(torch.bfloat16)

def process_data(args):
    """Main processing function (aligned with video version)"""
    # Define content and style restrictions
    content_restriction = ['perspective', 'multi_class_position', 'multi_class_appearance', 'instance']
    style_restriction = ['brief', 'detail', 'narrative', 'poem', 'theme']
    
    # Load model
    model, tokenizer = load_internvl_model(args.checkpoint, "cuda:0", args)
    device = model.device
    
    # Get image size configuration
    config = InternVLChatConfig.from_pretrained(args.checkpoint)
    image_size = config.force_image_size or config.vision_config.image_size
    
    # Generation configuration
    gen_config = {
        "do_sample": args.sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    # Read JSONL file
    with open(args.data_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    # Open output files
    with open(args.output_path_content, 'w', encoding='utf-8') as outfile_content, \
         open(args.output_path_style, 'w', encoding='utf-8') as outfile_style:
        
        for line in tqdm(lines, desc="Processing JSONL", unit="line"):
            if not line.strip():
                continue
                
            data = json.loads(line.strip())
            image_path = f"{args.image_dir}/{data['image']}"
            id = data['id']
            question = data['conversations'][0]['value']
            restriction = data['restriction'][0] if 'restriction' in data else 'detail'
            
            # Check if ID already exists
            if (is_id_exists(args.output_path_content, id) or 
               is_id_exists(args.output_path_style, id)):
                print(f"ID {id} already exists. Skipping...")
                continue
            
            try:
                # Load image
                pixel_values = load_image(image_path, image_size, device, args)
                
                # Generate response
                with torch.no_grad():
                    response = model.chat(tokenizer, pixel_values, question, gen_config)
                
                # Save results based on restriction type
                data_out = data.copy()
                
                if restriction in content_restriction:
                    data_out['model_response_content'] = response
                    outfile_content.write(json.dumps(data_out, ensure_ascii=False) + '\n')
                    outfile_content.flush()
                elif restriction in style_restriction:
                    data_out['model_response_style'] = response
                    outfile_style.write(json.dumps(data_out, ensure_ascii=False) + '\n')
                    outfile_style.flush()
                else:
                    print(f'Unknown restriction type: {restriction}')
                    
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Variable parameters


    parser.add_argument('--checkpoint', type=str, 
                       default='path/to/InternVL/model/checkpoint',
                       help='Model checkpoint path')
    parser.add_argument('--output_path_content', type=str, 
                       default='path/to/output/content.jsonl',
                       help='Path for model content outputs')
    parser.add_argument('--output_path_style', type=str, 
                       default='path/to/output/style.jsonl',
                       help='Path for model style outputs')
    parser.add_argument("--merged-output", type=str,
                       default='path/to/output/merged_results.jsonl',
                       help="Path for merged output file")
    
    # Fixed parameters  
    parser.add_argument('--data_path', type=str, 
                       default='path/to/data/anycapeval_image_ref.jsonl',
                       help='Path to the JSONL data file')
    parser.add_argument('--image_dir', type=str, 
                       default='path/to/test/image/directory',
                       help='Directory containing the images')

    parser.add_argument("--num-beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--sample", action="store_true", help="Enable random sampling")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic image processing")
    parser.add_argument("--max-num", type=int, default=6, help="Maximum number of image patches")
    parser.add_argument("--load-in-8bit", action="store_true", help="Enable 8-bit quantization")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantization")
    parser.add_argument("--auto", action="store_true", help="Enable automatic device mapping")
    args = parser.parse_args()

    # Create output directory if not exists
    os.makedirs(os.path.dirname(args.output_path_content), exist_ok=True)
    
    # Initialize empty output files
    for output_path in [args.output_path_content, args.output_path_style]:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("")

    process_data(args)
    
    # Merge and sort the output files
    merge_and_sort_outputs(args.output_path_content, args.output_path_style, args.merged_output)