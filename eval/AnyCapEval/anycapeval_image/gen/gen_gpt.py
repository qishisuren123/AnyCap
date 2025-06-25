import argparse
import json
import os
import base64
import requests
from tqdm import tqdm
from PIL import Image
import io
from typing import List, Dict, Any

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GPT-4o模型生成初始caption(图片模态)")
    
    # API configuration parameters
    parser.add_argument('--base_url', type=str, 
                        default="https://boyuerichdata.chatgptten.com/v1/chat/completions",
                        help='OpenAI API base URL')
    parser.add_argument('--api_key', type=str, 
                        default="sk-iproK7tAwu7J2ZBJWL8G3TiKUepPUH6uj5JQ7w0oXCRu02wl",
                        help='OpenAI API key')
    parser.add_argument('--model', type=str, default="gpt-4o-2024-08-06",
                        help='Model name to use')
    
    # Data path parameters
    parser.add_argument('--data_path', type=str, 
                       default='/mnt/petrelfs/renyiming/lzq_workspace/image_submit_code/anycapeval_image/anycapeval_image_ref.jsonl',
                       help='Path to the JSONL data file')
    parser.add_argument('--image_dir', type=str, 
                       default='/mnt/petrelfs/renyiming/lzq_workspace/QAC/test_QC/test_image',
                       help='Directory containing the images')
    parser.add_argument('--output_path_content', type=str, 
                       default='/mnt/petrelfs/renyiming/lzq_workspace/image_submit_code/anycapeval_image/output/temp_content_2.jsonl',
                       help='Path for content-related outputs')
    parser.add_argument('--output_path_style', type=str, 
                       default='/mnt/petrelfs/renyiming/lzq_workspace/image_submit_code/anycapeval_image/output/temp_style_2.jsonl',
                       help='Path for non-content-related outputs')
    parser.add_argument("--merged-output", type=str,
                       default='/mnt/petrelfs/renyiming/lzq_workspace/image_submit_code/anycapeval_image/output/merged_results_1.jsonl',
                       help="Path for merged and sorted output file")
    
    # Other parameters
    parser.add_argument('--max_tokens', type=int, default=2000,
                        help='Maximum tokens for API response')
    parser.add_argument('--max_retries', type=int, default=5,
                        help='Maximum retries for failed API calls')
    
    return parser.parse_args()

def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def create_gpt_payload(question: str, base64_image: str, model: str, max_tokens: int) -> Dict[str, Any]:
    """Create GPT API request payload with image"""
    content = [
        {"type": "text", "text": question},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": content}
    ]
    
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

def call_gpt_api(payload: Dict[str, Any], base_url: str, headers: Dict[str, str], max_retries: int) -> str:
    """Call GPT API using proxy service"""
    attempt = max_retries
    
    while attempt > 0:
        try:
            response = requests.post(base_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                response_json = response.json()
                return response_json["choices"][0]["message"]["content"].strip()
            else:
                print(f"API request failed, status code: {response.status_code}, response: {response.text}")
                attempt -= 1
        except Exception as e:
            attempt -= 1
            print("API call error:", e)
    
    raise Exception(f"API request failed after maximum retries")

def is_id_exists(file_path: str, id: int) -> bool:
    """Check if ID already exists in output file"""
    if not os.path.exists(file_path):
        return False
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            if data.get('id') == id:
                return True
    return False

def process_data(args):
    """Process data and perform image analysis"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}"  
    }
    
    # 定义内容类和非内容类限制
    content_restriction = ['perspective', 'multi_class_position', 'multi_class_appearance', 'instance']
    style_restriction = ['brief', 'detail', 'narrative', 'poem', 'theme']
    
    with open(args.data_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    with open(args.output_path_content, 'w', encoding='utf-8') as outfile_content, \
         open(args.output_path_style, 'w', encoding='utf-8') as outfile_style:
        
        for line in tqdm(lines, desc=f"Processing images with {args.model}", unit="image"):
            data = json.loads(line.strip())
            image_path = f"{args.image_dir}/{data['image']}"
            question = data['conversations'][0]['value']
            restriction = data['restriction'][0] if 'restriction' in data else 'detail'
            id = data.get('id')
            
            # 检查id是否已经存在
            if (os.path.exists(args.output_path_content) and is_id_exists(args.output_path_content, id)) or \
               (os.path.exists(args.output_path_style) and is_id_exists(args.output_path_style, id)):
                print(f"ID {id} already exists. Skipping...")
                continue
            
            try:
                base64_image = image_to_base64(image_path)
                
                payload = create_gpt_payload(question, base64_image, args.model, args.max_tokens)
                
                response = call_gpt_api(payload, args.base_url, headers, args.max_retries)
                
                result = data.copy()
                
                if restriction in content_restriction:
                    result['model_response_content'] = [response]
                    outfile_content.write(json.dumps(result, ensure_ascii=False) + '\n')
                    outfile_content.flush()
                elif restriction in style_restriction:
                    result['model_response_style'] = [response]
                    outfile_style.write(json.dumps(result, ensure_ascii=False) + '\n')
                    outfile_style.flush()
                else:
                    print(f'Restriction type "{restriction}" not in storage list, cannot process.')
                    
            except Exception as e:
                print(f"Error processing image {data['image']}: {str(e)}")
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

def main():
    """Main function"""
    args = parse_arguments()
    
    for output_path in [args.output_path_content, args.output_path_style, args.merged_output]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    print(f"Starting image analysis with {args.model}...")
    process_data(args)
    print(f"Processing completed. Content outputs saved to: {args.output_path_content}")
    print(f"Style outputs saved to: {args.output_path_style}")
    
    merge_and_sort_outputs(args.output_path_content, args.output_path_style, args.merged_output)

if __name__ == "__main__":
    main()