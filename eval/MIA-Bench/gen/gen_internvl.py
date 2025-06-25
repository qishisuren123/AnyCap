#For licensing see accompanying LICENSE file. Copyright (C) 2024 Apple Inc. All Rights Reserved.
import os
import pandas as pd
import json
from tqdm import tqdm
import torch
from PIL import Image
import torchvision.transforms as T
from urllib.parse import urlparse
# from dotenv import load_dotenv
from openai import OpenAI
import requests
from transformers import AutoModel, AutoTokenizer
import re
from urllib.parse import urlparse
import base64
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
# from openai import OpenAI
import time
import argparse

base_url = "https://boyuerichdata.chatgptten.com/v1/chat/completions"
skey = "sk-iproK7tAwu7J2ZBJWL8G3TiKUepPUH6uj5JQ7w0oXCRu02wl"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {skey}"  
}


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
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
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def create_json_output(image_url, instruction, caption):
    data = {"question_id": image_url, "prompt": instruction, "text": caption}
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))

def generate(input_file, output_file, model_path, image_folder):

    # 加载 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 加载模型和处理器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
        ).eval().cuda()


    # 初始化生成配置
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    results = []
    pattern = r"https?://[^\.]+\.([^\.]+)\."
    # 遍历数据并生成 caption
    for entry in data:
        img_path = entry["image"]
        parsed_url = urlparse(img_path)
        image_url_without_params = parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path

        url_parts = img_path.split('/')
        if img_path.endswith('.jpg'):
            if len(url_parts) >= 3:
                image_name = '/'.join(url_parts[-3:])  # 提取倒数第三个斜杠后的部分
            else:
                image_name = url_parts[-1]  # 如果 URL 中的 / 少于三个，就使用最后的部分
        else:
            image_name = img_path.split('/')[-1].split('?')[0]

        # 将URL中的斜杠替换为合法的文件名字符（如 `_`）
        image_name = image_name.replace('/', '_')

        if not image_name:
            match = re.search(pattern, img_path)
            domain_content = match.group(1) if match else None
            print('domain_content:' + domain_content)
            domain_content = domain_content + '.jpg'
            image_path = os.path.join(image_folder, domain_content)
            # print(image_path)
        else:
            image_path = os.path.join(image_folder, image_name)

        if not image_path.lower().endswith('.jpg') and not image_path.lower().endswith('.jpeg'):
            image_path += '.jpg'
        instruction = entry["instruction"]

        # 处理图片
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()


        # 使用模型生成回复
        response = model.chat(tokenizer, pixel_values, instruction, generation_config)

        # 单个模型的结果保存
        output = create_json_output(image_path, instruction, response)
        print(f"生成的描述: {response}")

        results.append(output)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for json_string in results:
            f.write(json_string + '\n')
        # json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"生成完成，结果保存到 {output_file}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Evaluate InternVL model in MIA-Bench.")
    parser.add_argument("--MIA_file", type=str, default="/mnt/petrelfs/renyiming/lzq_workspace/image_submit_code/MIA-Bench/instruction_benchmark_all.json", help="MIA-Bench's original file")
    parser.add_argument("--output_file", type=str, default="/mnt/petrelfs/renyiming/lzq_workspace/image_submit_code/MIA-Bench/output/inter_jsonl/InternVL2.5-8B_output.jsonl", help="Output path for evaluation results")
    parser.add_argument("--model_path", type=str, default="/mnt/petrelfs/share_data/wangweiyun/share_internvl/InternVL2_5-8B", help="Path to model")
    parser.add_argument("--image_folder", type=str, default="/mnt/petrelfs/renyiming/lzq_workspace/image_submit_code/MIA-Bench/image", help="Path to model")

    args = parser.parse_args()

    # # internvl模型生成caption
    generate(input_file = args.MIA_file, output_file = args.output_file, model_path = args.model_path, image_folder = args.image_folder)

