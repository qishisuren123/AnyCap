import os
import json
from urllib.parse import urlparse
import requests
import re
from urllib.parse import urlparse
import base64
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


def create_json_output(image_url, instruction, caption):
    data = {"question_id": image_url, "prompt": instruction, "text": caption}
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def gpt(modified_prompt, base64_image):
    attempt = 5

    while attempt > 0:
        try:
            data = {
                        "model":"gemini-1.5-pro",
                        "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": modified_prompt},
                                            {"type": "image_url",
                                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                            },
                                        ],
                                    }
                                ],
                                "max_tokens":2000
                    }
            response = requests.post(base_url, headers=headers, json=data)
            response_gpt = response.json()['choices'][0]['message']['content']

            print(response_gpt)
           
            return response_gpt

        except Exception as e:
            attempt -= 1
            print("Error occurred:", e)
            print(f"response.status_code: {response.status_code}")
            time.sleep(1)
    print('调用api出现异常，无法得到正常返回结果。')
    return 'None.'



def generate(input_file, output_file):

    # 加载 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)


    results = []
    image_folder = "/mnt/petrelfs/renyiming/lzq_workspace/MIA-Bench/image"
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
        base64_image = encode_image(image_path)

        # gpt返回回答
        response = gpt(instruction, base64_image)

        output = create_json_output(image_path, instruction, response)
        print(f"生成的描述: {response}")

        results.append(output)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for json_string in results:
            f.write(json_string + '\n')

    print(f"生成完成，结果保存到 {output_file}")



if __name__ == '__main__':
    # print('v和i联合起来训练了一个epoch')
    parser = argparse.ArgumentParser(description = "Evaluate GPT-4o+8b aligner model in MIA-Bench.")
    parser.add_argument("--MIA_file", type=str, default="/image_submit_code/MIA-Bench/instruction_benchmark_all.json", help="MIA-Bench's original file")
    parser.add_argument("--output_file", type=str, default="/image_submit_code/MIA-Bench/inter_jsonl/model_test_result/gemini-1.5-pro_output.jsonl", help="Output path for results")
    parser.add_argument("--image_folder", type=str, default="image_submit_code/MIA-Bench/image", help="Path to model")

    args = parser.parse_args()

    # 生成caption
    generate(input_file = args.MIA_file, output_file = args.output_file, image_folder = args.image_folder)
    

