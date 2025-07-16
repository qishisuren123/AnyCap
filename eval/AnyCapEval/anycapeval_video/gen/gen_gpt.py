import argparse
import json
import torch
from PIL import Image
from transformers import AutoTokenizer
import base64
import requests
from tqdm import tqdm
from decord import VideoReader, cpu
import numpy as np
import os
import io
from typing import List, Dict, Any, Tuple

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # API configuration parameters
    parser.add_argument('--base_url', type=str, 
                        default="https://boyuerichdata.chatgptten.com/v1/chat/completions",
                        help='OpenAI API base URL')
    parser.add_argument('--api_key', type=str, 
                        default="your/api/key",
                        help='OpenAI API key')
    parser.add_argument('--model', type=str, default="gpt-4o-2024-08-06",
                        help='Model name to use, e.g.: gpt-4o, gpt-4o-mini, gpt-4-turbo, claude-3-5-sonnet-20240620')
    
    # Data path parameters
    parser.add_argument('--data_path', type=str, 
                       default='/path/to/anycapeval_video_ref.jsonl',
                       help='Path to the JSONL data file')
    parser.add_argument('--video_dir', type=str, 
                       default='path/to/test/video/directory',
                       help='Directory containing the video files')
    parser.add_argument('--output_path_content', type=str, default='/path/to/output/temp_content.jsonl',
                       help='Path for content-related outputs')
    parser.add_argument('--output_path_style', type=str, default='/path/to/output/temp_style.jsonl',
                       help='Path for style-related outputs')
    parser.add_argument("--merged-output", type=str,
                       default='/path/to/output/merged_results.jsonl',
                       help="Path for merged output file")
    
    # Other parameters
    parser.add_argument('--num_segments', type=int, default=8, 
                        help='Number of frames to extract from each video')
    parser.add_argument('--max_tokens', type=int, default=1000,
                        help='Maximum tokens for API response')
    parser.add_argument('--max_retries', type=int, default=5,
                        help='Maximum retries for failed API calls')
    
    return parser.parse_args()

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

def extract_video_frames(video_path: str, num_segments: int = 16) -> List[Image.Image]:
    """Extract frames from video"""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    
    frame_indices = get_index(None, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    frames = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        frames.append(img)
    
    return frames

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def create_gpt_payload(question: str, frames: List[Image.Image], model: str, max_tokens: int) -> Dict[str, Any]:
    """Create GPT API request payload with images"""
    content = [{"type": "text", "text": question}]
    
    for i, frame in enumerate(frames):
        base64_image = image_to_base64(frame)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
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

def process_data(args):
    """Process data and perform video analysis"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}"  
    }
    
    with open(args.data_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    with open(args.output_path_content, 'w', encoding='utf-8') as outfile_content, \
         open(args.output_path_style, 'w', encoding='utf-8') as outfile_style:
        
        for line in tqdm(lines, desc=f"Processing videos with {args.model}", unit="video"):
            data = json.loads(line.strip())
            
            video_path = f"{args.video_dir}/{data['video']}"
            question = data['conversations'][0]['value']
            restriction = data['restriction'][0] if 'restriction' in data else 'detail'
            
            try:
                frames = extract_video_frames(video_path, num_segments=args.num_segments)
                
                payload = create_gpt_payload(question, frames, args.model, args.max_tokens)
                
                response = call_gpt_api(payload, args.base_url, headers, args.max_retries)
                
                content_restriction = ['action', 'appearance', 'background', 'camera', 'event', 'instance', 'perspective', 'position']
                style_restriction = ['brief', 'detail', 'narrative', 'poem', 'theme']
                
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
                print(f"Error processing video {data['video']}: {str(e)}")
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

def main():
    """Main function"""
    args = parse_arguments()
    
    for output_path in [args.output_path_content, args.output_path_style, args.merged_output]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    print(f"Starting video analysis with {args.model}...")
    process_data(args)
    print(f"Processing completed. Content outputs saved to: {args.output_path_content}")
    print(f"Style outputs saved to: {args.output_path_style}")
    
    merge_and_sort_outputs(args.output_path_content, args.output_path_style, args.merged_output)
if __name__ == "__main__":
    main()