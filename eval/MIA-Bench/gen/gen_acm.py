import os
import json
import torch
from PIL import Image
import torchvision.transforms as T
from urllib.parse import urlparse
import re
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# 图像处理相关函数保持不变
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    return torch.stack(pixel_values)

def create_json_output(image_url, instruction, caption):
    return json.dumps({"question_id": image_url, "prompt": instruction, "text": caption}, 
                     ensure_ascii=False, separators=(',', ':'))

def generate_aligner_captions(input_file, output_file, model_path, image_folder):

    tokenizer_aligner = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model_aligner = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
        ).eval().cuda()
    
    # 加载JSONL数据（逐行读取）
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]  # 读取 JSONL 文件
    
    results = []
    pattern = r"https?://[^\.]+\.([^\.]+)\."
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    
    for entry in tqdm(data, desc="Generating aligner captions"):
        image_path = entry["question_id"]
        
        instruction = entry["prompt"]
        
        # 加载图像
        pixel_values = load_image(image_path).to(torch.bfloat16).cuda()
        
        # 生成aligner提示
        prompt_aligner = f"<image>\nYou are a multimodal aligner, and your task is to transform the existing caption into a caption that better meets the requirements. The requirement is: '{instruction}'. The existing caption is: '{entry['text']}'. You only need to answer the caption that better meets the requirements, and do not add anything else."
        
        # 生成aligner响应
        response_aligner = model_aligner.chat(
            tokenizer_aligner, 
            pixel_values, 
            prompt_aligner, 
            generation_config
        )
        
        # 保存结果
        output = create_json_output(image_path, instruction, response_aligner)
        results.append(output)
    
    # 写入结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for json_string in results:
            f.write(json_string + '\n')
    
    print(f"Aligner captions generated and saved to {output_file}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate aligner captions using InternVL-8B model")
    parser.add_argument("--input_file", type=str,
                        default="/image_submit_code/MIA-Bench/output/inter_jsonl/InternVL2.5-8B_output.jsonl",
                        help="Input JSON file path")
    parser.add_argument("--output_file", type=str,
                        default="/image_submit_code/MIA-Bench/output/inter_jsonl/acm_output.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--model_path", type=str,
                        default="ACM_model_path",
                        help="Path to ACM")
    parser.add_argument("--image_folder", type=str,
                        default="/image_submit_code/MIA-Bench/image",
                        help="Path to image folder")
    
    args = parser.parse_args()
    generate_aligner_captions(args.input_file, args.output_file, args.model_path, args.image_folder)