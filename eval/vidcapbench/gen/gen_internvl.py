import argparse
import os
import json
import sys

from tqdm import tqdm
import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from decord import VideoReader, cpu
import numpy as np


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
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


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
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


def process_videos(args):
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args)
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")

    generation_config = dict(
        do_sample=args.sample,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    question = "Describe the video in detail."

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = open(os.path.join(args.output_dir, args.output_name), 'w', encoding='utf-8')

    video_files = [f for f in os.listdir(args.video_dir)
                  if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    batch_size = 1  
    for i in tqdm(range(0, len(video_files), batch_size), desc="Processing video batches"):
        batch_video_files = video_files[i:i + batch_size]

        pixel_values_batch = []
        all_num_patches_lists = []  
        valid_video_files = []  

        for video_file in batch_video_files:
            try:
                video_path = os.path.join(args.video_dir, video_file)
                pixel_values, num_patches_list = load_video(
                    video_path,
                    num_segments=8,
                    max_num=1
                )
                pixel_values_batch.append(pixel_values)
                all_num_patches_lists.append(num_patches_list)
                valid_video_files.append(video_file)
            except Exception as e:
                print(f"Error processing video {video_file}: {str(e)}")
                continue

        if not pixel_values_batch:
            continue

        pixel_values = torch.cat(pixel_values_batch, dim=0).cuda().to(torch.bfloat16)

        full_questions = []
        for num_patches_list in all_num_patches_lists:
            video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))])
            full_questions.append(video_prefix + question)

        responses = model.batch_chat_video(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            questions=full_questions,
            generation_config=generation_config,
            num_patches_list=all_num_patches_lists,
        )

        for video_file, response in zip(valid_video_files, responses):
            result = {
                "question": "Describe the video in detail.",
                "video_dir": video_file,
                "model_generation": response.strip()
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            output_file.flush()

    output_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='/mnt/petrelfs/share_data/wangweiyun/share_internvl/InternVL2_5-8B')
                        
    parser.add_argument('--video-dir', type=str,
                        default='/mnt/petrelfs/renyiming/ly_workspace/video_benchs/VidCapBench/videos',
                        help='Directory containing video files')
    parser.add_argument('--output-dir', type=str, default='/mnt/petrelfs/renyiming/ly_workspace/submit_code/vidcapbench/output',
                        help='Directory to save output files')
    parser.add_argument('--output-name', type=str, default='internvl2.5-8b.jsonl',
                        help='Directory to save output files')

    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')

    args = parser.parse_args()
    process_videos(args)


if __name__ == '__main__':
    main()