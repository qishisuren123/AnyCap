import argparse
import os
import json
import base64
import io
from typing import List, Dict, Any

from tqdm import tqdm
import requests
from PIL import Image
from decord import VideoReader, cpu
import numpy as np


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """Calculate frame indices for video sampling"""
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


def extract_frames(video_path, num_segments=8):
    """Extract frames from video file"""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    
    frame_indices = get_index(None, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    frames = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        frames.append(img)
    
    return frames


def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_gpt4o_payload(question, frames, max_tokens=1000):
    """Create payload for GPT-4o API with images"""
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
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": max_tokens,
    }


def call_gpt4o_api(payload, base_url, headers, max_retries=5):
    """Call GPT-4o API with retry mechanism"""
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
    
    raise Exception(f"API request failed after {max_retries} attempts")


def process_videos(args):
    """Process videos using GPT-4o API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}"
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = open(os.path.join(args.output_dir, args.output_name), 'w', encoding='utf-8')
    
    video_files = [f for f in os.listdir(args.video_dir)
                  if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

                  
    for video_file in tqdm(video_files, desc="Processing videos"):
        try:
            video_path = os.path.join(args.video_dir, video_file)
            
            frames = extract_frames(video_path, num_segments=args.num_segments)
            
            payload = create_gpt4o_payload(
                question=args.question,
                frames=frames,
                max_tokens=args.max_tokens
            )
            
            response = call_gpt4o_api(
                payload=payload,
                base_url=args.base_url,
                headers=headers,
                max_retries=args.max_retries
            )
            
            result = {
                "question": args.question,
                "video_dir": video_file,
                "model_generation": response.strip()
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            output_file.flush()
            
        except Exception as e:
            print(f"Error processing video {video_file}: {str(e)}")
            error_result = {
                "question": args.question,
                "video_dir": video_file,
                "error": str(e)
            }
            output_file.write(json.dumps(error_result, ensure_ascii=False) + '\n')
            output_file.flush()
    
    output_file.close()


def main():
    parser = argparse.ArgumentParser(description="VidCapBench with GPT-4o")
    
    # API configuration parameters
    parser.add_argument('--base_url', type=str, 
                      default="https://boyuerichdata.chatgptten.com/v1/chat/completions",
                      help='OpenAI API base URL')
    parser.add_argument('--api_key', type=str, 
                      default='',
                      help='OpenAI API key')
    parser.add_argument('--model', type=str, default="gpt-4o-2024-08-06",
                      help='Model name to use')
    
    # Data path parameters
    parser.add_argument('--video-dir', type=str,
                        default='/VidCapBench/videos',
                        help='Directory containing video files')
    parser.add_argument('--output-dir', type=str, default='vidcapbench/output',
                        help='Directory to save output files')
    parser.add_argument('--output-name', type=str, default='gpt-4o-2024-08-06.jsonl',
                        help='Directory to save output files')
    
    # Processing parameters
    parser.add_argument('--question', type=str, 
                      default="Describe the video in detail.",
                      help='Question to ask about the video')
    parser.add_argument('--num-segments', type=int, default=8,
                      help='Number of frames to extract from each video')
    parser.add_argument('--max-tokens', type=int, default=1000,
                      help='Maximum tokens for API response')
    parser.add_argument('--max-retries', type=int, default=5,
                      help='Maximum number of API call retries')
    
    args = parser.parse_args()
    process_videos(args)


if __name__ == '__main__':
    main()