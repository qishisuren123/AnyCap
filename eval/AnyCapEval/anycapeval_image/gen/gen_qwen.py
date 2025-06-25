import argparse
import json
import os
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import torch

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
    
    print(f"Merged {len(all_data)} records and saved to {output_file}")

def load_qwen_model(checkpoint, device):
    """Load Qwen model and processor"""
    device_map = {"": device}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map=device_map
    ).eval()
    
    processor = AutoProcessor.from_pretrained(checkpoint)
    
    return model, processor

def process_conversation(model, processor, image_path, question):
    """Process a single conversation"""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    
    return response

def process_data(args):
    """Main processing function"""
    # Define content and non-content restrictions
    content_restriction = ['perspective', 'multi_class_position', 'multi_class_appearance', 'instance']
    style_restriction = ['brief', 'detail', 'narrative', 'poem', 'theme']
    
    # Load model
    model, processor = load_qwen_model(args.checkpoint, "cuda:0")
    
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
            restriction = data['restriction'][0]
            
            # Check if ID already exists
            if (is_id_exists(args.output_path_content, id) or 
               is_id_exists(args.output_path_style, id)):
                print(f"ID {id} already exists. Skipping...")
                continue
            
            try:
                # Generate response
                response = process_conversation(model, processor, image_path, question)
                
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
    parser = argparse.ArgumentParser(description="Qwen model for caption generation")
    # Variable parameters
    parser.add_argument('--checkpoint', type=str,
                       default='/mnt/petrelfs/renyiming/lzq_workspace/new_bench/checkpoint/Qwen2.5-VL-7B-Instruct',
                       help='Model checkpoint path')
    parser.add_argument('--output_path_content', type=str,
                       default='/mnt/petrelfs/renyiming/lzq_workspace/image_submit_code/anycapeval_image/output/qwen_content.jsonl',
                       help='Path for content outputs')
    parser.add_argument('--output_path_style', type=str,
                       default='/mnt/petrelfs/renyiming/lzq_workspace/image_submit_code/anycapeval_image/output/qwen_style.jsonl',
                       help='Path for style outputs')
    parser.add_argument("--merged_output", type=str,
                       default='/mnt/petrelfs/renyiming/lzq_workspace/image_submit_code/anycapeval_image/output/merged_results.jsonl',
                       help="Path for merged and sorted output file")
    
    # Fixed parameters
    parser.add_argument('--data_path', type=str,
                       default='/mnt/petrelfs/renyiming/lzq_workspace/image_submit_code/anycapeval_image/anycapeval_image_ref.jsonl',
                       help='Path to the JSONL data file')
    parser.add_argument('--image_dir', type=str,
                       default='/mnt/petrelfs/renyiming/lzq_workspace/image_submit_code/anycapeval_image/test_image_data',
                       help='Directory containing the images')
    
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