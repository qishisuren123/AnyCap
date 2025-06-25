# import json
# import base64
# import requests
# from tqdm import tqdm
# from openai import OpenAI
# import argparse
# import re
# import time
# import os

# # 新的直连key
# client = OpenAI(
#     api_key='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',  # This is the default and can be omitted
# )

# def validate_response_format(response):
#     try:
#         # 检查顶级键
#         assert "caption_evaluation" in response, "Missing 'caption_evaluation'"

#         # 定义一个内部函数来检查模型评估部分
#         def check_evaluation(evaluation, model_name):
#             assert "key_points_scores" in evaluation, f"Missing 'key_points_scores' in {model_name}"
#             assert "total_score" in evaluation, f"Missing 'total_score' in {model_name}"
#             assert "score_reasons" in evaluation, f"Missing 'score_reasons' in {model_name}"

#             # 检查 key_points_scores 结构
#             assert isinstance(evaluation["key_points_scores"], dict), f"'key_points_scores' in {model_name} must be a dict"
#             for key_point, score in evaluation["key_points_scores"].items():
#                 assert isinstance(key_point, str), f"Key point '{key_point}' in {model_name} must be a string"
#                 assert isinstance(score, int), f"Score for '{key_point}' in {model_name} must be an integer"

#             # 检查 total_score 结构
#             assert isinstance(evaluation["total_score"], int), f"'total_score' in {model_name} must be an integer"

#             # 检查 score_reasons 结构
#             assert isinstance(evaluation["score_reasons"], dict), f"'score_reasons' in {model_name} must be a dict"
#             for key_point, reason in evaluation["score_reasons"].items():
#                 assert isinstance(key_point, str), f"Key point '{key_point}' in {model_name} must be a string"
#                 assert isinstance(reason, str), f"Reason for '{key_point}' in {model_name} must be a string"

#             # 检查 key_points_scores 和 score_reasons 的键是否一致
#             assert set(evaluation["key_points_scores"].keys()) == set(evaluation["score_reasons"].keys()), \
#                 f"Keys in 'key_points_scores' and 'score_reasons' do not match in {model_name}"

#             # 检查总分是否正确
#             total_score_calculated = sum(evaluation["key_points_scores"].values())
#             assert evaluation["total_score"] == total_score_calculated, \
#                 f"'total_score' in {model_name} is {evaluation['total_score']}, but calculated score is {total_score_calculated}"

#         # 检查两个模型的评估
#         check_evaluation(response["caption_evaluation"], "caption_evaluation")

#         return True
#     except AssertionError as e:
#         print(f"Validation failed: {e}")
#         return False

# def gpt_reference_content(modified_prompt):
#     attempt = 5

#     while attempt > 0:
#         try:
#             # 直连api获得response
#             completion = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[{
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text", 
#                             "text": modified_prompt
#                         }
#                     ]
#                 }]
#             )
#             response = completion.choices[0].message.content.strip()
#             # print(response)

#             try:
#                 response_data = json.loads(response)
#                 print(response_data)
#                 if validate_response_format(response_data):
#                     print('调用api成功，正常返回结果。')
#                     return response
#                 else:
#                     print('返回结果格式无效，再次尝试。')
#                     attempt -= 1
#             except Exception as e:
#                 print('返回结果无效，再次尝试。')
#                 attempt -= 1
#                 print("Error occurred:", e)
#                 time.sleep(2)
#         except Exception as e:
#             attempt -= 1
#             print("Error occurred:", e)
#             # print(f"Request failed with status code {response.status_code}")
#             time.sleep(2)
#     print('调用api出现异常，无法得到正常返回结果。')
#     return '{"requirements_extracted": "error", "requirement_scores": "error", "total_score": "error", "score_reasons": "reason"}'


# def encode_audio(audio_path):
#     with open(audio_path, "rb") as f:
#         return base64.b64encode(f.read()).decode('utf-8')


# # 评测全部数据时使用的函数
# def load_new_data_ids(new_data_path):
#     """读取新的jsonl文件，并提取需要评测的ID集合"""
#     new_data_ids = set()
#     new_data_dict = {}
#     with open(new_data_path, 'r', encoding='utf-8') as new_file:
#         for line in new_file:
#             data = json.loads(line)
#             new_data_ids.add(data['id'])
#             new_data_dict[data['id']] = data  # 存储完整数据，便于后续查找
#     return new_data_ids, new_data_dict


# def main(data_path, content_template_path, audio_dir, output_path, test_data_path):
#     # 读取数据中的 ID
#     new_data_ids, new_data_dict = load_new_data_ids(test_data_path)

#     # 读取需要处理的数据文件
#     data_list = []
#     with open(data_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             data = json.loads(line)
#             data_list.append(data)

#     print(f"找到 {len(data_list)} 行数据需要处理")

#     # 读取内容类 prompt 模板
#     with open(content_template_path, 'r', encoding='utf-8') as file:
#         prompt_gpt_content = file.read()

#     # 生成答案并输出
#     with open(output_path, 'w', encoding='utf-8') as output_file:
#         print(f'已打开输出文件。{output_path}')

#         for data in tqdm(data_list, desc='Processing data'):
#             id_ = data['id']
#             if id_ not in new_data_ids:
#                 # 如果该 ID 不在新数据中，则跳过
#                 continue

#             new_data = new_data_dict[id_]

#             audio_path = f"{audio_dir}/{data['audio']}"  # 拼接图像路径
#             answer = data['model_response_content']
#             restriction = data['restriction'][0]

#             # 提取关键点
#             key_points = new_data.get('key_point', [])
#             # 将 key_points 列表转换为字符串，每个关键点占一行
#             key_points_str = "\n".join(key_points)

#             modified_prompt = prompt_gpt_content.replace("{key_points}", key_points_str).replace("{answer}", answer)
#             response_gpt = gpt_reference_content(modified_prompt)
#             response_gpt_dict = json.loads(response_gpt)
#             caption_eval = response_gpt_dict.get("caption_evaluation", {})

#             # 将结果添加回 new_data 中
#             new_data["model_evaluation"] = caption_eval
#             new_data["model_caption"] = answer
#             output_file.write(json.dumps(new_data, ensure_ascii=False) + '\n')
#             output_file.flush()

#     print('已完成评测，开始计算得分。')

# def sentences_count(text):
#     words = text.split()
#     word_count = len(words)

#     sentences = re.split(r'[.!?]', text)  # 以标点分割句子
#     sentence_count = sum(1 for s in sentences if s.strip())  # 计算非空句子

#     return sentence_count

# def count_total_words(mixed_str):
#     # 统计英文单词（包括带撇号和连字符的单词）
#     en_words = re.findall(r"\b[a-zA-Z]+(?:['-][a-zA-Z]+)*\b", mixed_str)
    
#     # 统计中文字符（包括中文标点）
#     zh_chars = re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", mixed_str)
    
#     # 返回英文单词数和中文字数的总和
#     return len(en_words) + len(zh_chars)


# def sentence_modify(data, limit, caption, model_type):

#     model_sentence_length = sentences_count(caption)

#     if limit == "The generated caption's length needs to be one sentence.":
#         if model_sentence_length == 1:
#             return True
#         else:
#             print(f"模型为{model_type},失败的限制为：{limit},{data['restriction'][0]}")
#             return False

#     if limit == "The generated caption's length needs to be exactly two sentences.":
#         if model_sentence_length == 2:
#             return True
#         else:
#             print(f"模型为{model_type},失败的限制为：{limit},{data['restriction'][0]}")
#             return False

#     if limit == "The generated caption's length cannot exceed two sentences.":
#         if model_sentence_length <= 2:
#             return True
#         else:
#             print(f"模型为{model_type},失败的限制为：{limit},{data['restriction'][0]}")
#             return False

#     if limit == "The generated caption's length needs to be exactly three sentences.":
#         if model_sentence_length == 3:
#             return True
#         else:
#             print(f"模型为{model_type},失败的限制为：{limit},{data['restriction'][0]}")
#             return False

#     if limit == "The generated caption's length cannot exceed three sentences.":
#         if model_sentence_length <= 3:
#             return True
#         else:
#             print(f"模型为{model_type},失败的限制为：{limit},{data['restriction'][0]}")
#             return False
    
#     if limit == "The generated caption's length needs to be exactly five sentences.":
#         if model_sentence_length == 5:
#             return True
#         else:
#             print(f"模型为{model_type},失败的限制为：{limit},{data['restriction'][0]}")
#             return False


# def word_modify(data, limit, caption, model_type):

#     words_model = count_total_words(caption)

#     if limit == "The generated caption's length needs to be no more than 10 words.":
#         if words_model <= 10:
#             return True
#         else:
#             print(f"模型为{model_type},失败的限制为：{limit}, Restriction为{data['restriction'][0]}, 实际单词数量为{words_model} words. ID为：{data['id']}")
#             return False
    
#     if limit == "The generated caption's length needs to be exactly 10 words.":
#         if words_model == 10:
#             return True
#         else:
#             print(f"模型为{model_type},失败的限制为：{limit}, Restriction为{data['restriction'][0]}, 实际单词数量为{words_model} words. ID为：{data['id']}")
#             return False
    
#     if limit == "The generated caption's length needs to be 10 to 20 words.":
#         if 10 <= words_model <= 20:
#             return True
#         else:
#             print(f"模型为{model_type},失败的限制为：{limit}, Restriction为{data['restriction'][0]}, 实际单词数量为{words_model} words. ID为：{data['id']}")
#             return False
    
#     if limit == "The generated caption's length needs to be exactly 20 words.":
#         if words_model == 20:
#             return True
#         else:
#             print(f"模型为{model_type},失败的限制为：{limit}, Restriction为{data['restriction'][0]}, 实际单词数量为{words_model} words. ID为：{data['id']}")
#             return False
    
#     if limit == "The generated caption's length needs to be 20 to 30 words.":
#         if 20 <= words_model <= 30:
#             return True
#         else:
#             print(f"模型为{model_type},失败的限制为：{limit}, Restriction为{data['restriction'][0]}, 实际单词数量为{words_model} words. ID为：{data['id']}")
#             return False
    
#     if limit == "The generated caption's length needs to be exactly 50 words.":
#         if words_model == 50:
#             return True
#         else:
#             print(f"模型为{model_type},失败的限制为：{limit}, Restriction为{data['restriction'][0]}, 实际单词数量为{words_model} words. ID为：{data['id']}")
#             return False
    
#     if limit == "The generated caption's length needs to be exactly 60 words.":
#         if words_model == 60:
#             return True
#         else:
#             print(f"模型为{model_type},失败的限制为：{limit}, Restriction为{data['restriction'][0]}, 实际单词数量为{words_model} words. ID为：{data['id']}")
#             return False
    
#     if limit == "The generated caption's length needs to be 30 to 120 words.":
#         if 30 <= words_model <= 120:
#             return True
#         else:
#             print(f"模型为{model_type},失败的限制为：{limit}, Restriction为{data['restriction'][0]}, 实际单词数量为{words_model} words. ID为：{data['id']}")
#             return False
    


# def modify_score(data, model_type):

#     caption = data['model_caption']
#     # print(data['restriction'])
#     len_limit = data['restriction'][1]

#     if "sentence" in len_limit:
#         return sentence_modify(data, len_limit, caption, model_type)
#     elif "word" in len_limit:
#         return word_modify(data, len_limit, caption, model_type)

#     print('出现问题，限制里不含sentence或word')
#     return True


# def calculate_model_scores(output_path):
#     # 初始化统计变量（惩罚前）
#     pre_model_total_score = 0
#     pre_model_max_score = 0
    
#     # 初始化统计变量（惩罚后）
#     post_model_total_score = 0
#     post_model_max_score = 0
    
#     data_count = 0
#     model_total_words = 0

#     path_dir = os.path.dirname(output_path)
#     file_name = 'modified' + os.path.basename(output_path)
#     modify_score_path = os.path.join(path_dir, file_name)

#     with open(output_path, 'r') as file1, open(modify_score_path, 'w') as file2:
#         for line in file1:
#             data = json.loads(line)

#             # 检查 caption1_evaluation 是否存在
#             if 'model_evaluation' not in data:
#                 continue
            
#             # 检查 total_score 是否存在
#             if 'total_score' not in data['model_evaluation']:
#                 continue
            
#             # 计算当前行的理论最大得分
#             max_score_per_line = len(data['key_point'])
            
#             # 惩罚前的分数
#             pre_model_score = data['model_evaluation']['total_score']
            
#             # 惩罚前的统计
#             pre_model_total_score += pre_model_score
#             pre_model_max_score += max_score_per_line

#             # 统计单词数量
#             if 'model_caption' in data:
#                 caption = data['model_caption']
#                 # 使用正则表达式统计单词（考虑英文单词边界）
#                 words = re.findall(r'\b\w+(?:-\w+)*\b', caption)  # 支持带连字符的单词
#                 model_total_words += len(words)
#             else:
#                 print(f"Warning: Missing 'model_caption' in ID {data.get('id', 'unknown')}")
            
#             # 添加长度惩罚项
#             model_score = pre_model_score
#             if not modify_score(data=data, model_type='model1'): # 长度不符合要求，施加惩罚
#                 model_score -= 1
#                 data['model_evaluation']['total_score'] = max(model_score, 0)
#                 print(f"ID为{data['id']}，Model1长度不符合要求扣一分。")
#                 data_count += 1
            
#             # 惩罚后的统计
#             post_model_total_score += model_score
#             post_model_max_score += max_score_per_line
            
#             file2.write(json.dumps(data, ensure_ascii=False) + '\n')
#             file2.flush()

#     # 计算得分百分比
#     pre_model_percentage = (pre_model_total_score / pre_model_max_score) * 100
#     post_model_percentage = (post_model_total_score / post_model_max_score) * 100

#     # 计算得分密度（每100单词得分）
#     if model_total_words > 0:
#         pre_score_density = (pre_model_total_score / model_total_words) * 100
#     else:
#         pre_score_density = 0
#     if model_total_words > 0:
#         post_score_density = (post_model_total_score / model_total_words) * 100
#     else:
#         post_score_density = 0

#     # 输出惩罚前的结果
#     print("\n=== 惩罚前统计 ===")
#     print(f"Model1 的总得分: {pre_model_total_score}, 得分百分比: {pre_model_percentage:.2f}%")
#     print(f"得分密度（每100单词得分）为: {pre_score_density:.2f}")

#     # 输出惩罚后的结果
#     print("\n=== 惩罚后统计 ===")
#     print(f"Model1 的总得分: {post_model_total_score}, 得分百分比: {post_model_percentage:.2f}%")
#     print(f"得分密度（每100单词得分）为: {post_score_density:.2f}")
#     print(f"有 {data_count} 个样本因长度限制被扣分。")

#     # 结果输出到文件
#     summary_path = output_path.replace(".jsonl", "_summary.txt")
#     with open(summary_path, 'w', encoding='utf-8') as summary_file:
#         summary_file.write(f"评测模型为：{output_path}\n\n")
#         summary_file.write("=== 惩罚前统计 ===\n")
#         summary_file.write(f"Model1 的总得分: {pre_model_total_score}, 得分百分比: {pre_model_percentage:.2f}%\n")
#         summary_file.write(f"得分密度（每100单词得分）为: {pre_score_density:.2f}\n\n")

#         summary_file.write("=== 惩罚后统计 ===\n")
#         summary_file.write(f"Model1 的总得分: {post_model_total_score}, 得分百分比: {post_model_percentage:.2f}%\n")
#         summary_file.write(f"得分密度（每100单词得分）为: {post_score_density:.2f}\n")
#         summary_file.write(f"有 {data_count} 个样本因长度限制被扣分。\n")

#     print(f"\n已将评分统计结果保存到文件：{summary_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_path", type=str, default='', help="Path to model output JSONL")
#     parser.add_argument("--content_template_path", type=str, default='', help="Prompt template for GPT input")
#     parser.add_argument("--audio_dir", type=str, default='', help="Directory containing audio files")
#     parser.add_argument("--output_path", type=str, default='', help="Output path for data_path_2 evaluated results")
#     parser.add_argument("--test_data_path", type=str, default='', help="Test data path with keypoints")

#     args = parser.parse_args()

#     # 生成评估结果
#     main(args.data_path, args.content_template_path, args.audio_dir, args.output_path, args.test_data_path)

#     # 得到评估得分
#     calculate_model_scores(args.output_path)


import json
import base64
import requests
from tqdm import tqdm
from openai import OpenAI
import argparse
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置常量
DEFAULT_MAX_RETRIES = 5
DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_WORKERS = 4

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Audio caption quality assessment tool')
    
    # API configuration parameters
    parser.add_argument('--api_key', type=str, default='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',
                        help='OpenAI API key')
    parser.add_argument('--model', type=str, default="gpt-4o",
                        help='Model name to use')
    
    # Data path parameters
    parser.add_argument('--data_path', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/eval/input/content_input/test.jsonl',
                        help='Original model result file path')
    parser.add_argument('--output_path', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/eval/output/test_content.jsonl',
                        help='Evaluation result output file path')
    parser.add_argument('--content_template_path', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/instruction_content.txt',
                        help='Prompt template file path')
    parser.add_argument('--test_data_path', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/anycapeval_audio_ref.jsonl',
                        help='Test data path with keypoints')
    
    # Processing parameters
    parser.add_argument('--max_retries', type=int, default=DEFAULT_MAX_RETRIES,
                        help='Maximum API call retry attempts')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for processing')
    parser.add_argument('--max_workers', type=int, default=DEFAULT_MAX_WORKERS,
                        help='Number of worker threads in thread pool')
    parser.add_argument('--calculate_only', action='store_true',
                        help='Only calculate score statistics for existing evaluation results')
    
    return parser.parse_args()

def validate_response_format(response):
    """Validate the structure of the API response"""
    try:
        assert "caption_evaluation" in response, "Missing 'caption_evaluation'"

        def check_evaluation(evaluation, model_name):
            assert "key_points_scores" in evaluation, f"Missing 'key_points_scores' in {model_name}"
            assert "total_score" in evaluation, f"Missing 'total_score' in {model_name}"
            assert "score_reasons" in evaluation, f"Missing 'score_reasons' in {model_name}"

            assert isinstance(evaluation["key_points_scores"], dict), f"'key_points_scores' in {model_name} must be a dict"
            for key_point, score in evaluation["key_points_scores"].items():
                assert isinstance(key_point, str), f"Key point '{key_point}' in {model_name} must be a string"
                assert isinstance(score, int), f"Score for '{key_point}' in {model_name} must be an integer"

            assert isinstance(evaluation["total_score"], int), f"'total_score' in {model_name} must be an integer"

            assert isinstance(evaluation["score_reasons"], dict), f"'score_reasons' in {model_name} must be a dict"
            for key_point, reason in evaluation["score_reasons"].items():
                assert isinstance(key_point, str), f"Key point '{key_point}' in {model_name} must be a string"
                assert isinstance(reason, str), f"Reason for '{key_point}' in {model_name} must be a string"

            assert set(evaluation["key_points_scores"].keys()) == set(evaluation["score_reasons"].keys()), \
                f"Keys in 'key_points_scores' and 'score_reasons' do not match in {model_name}"

            total_score_calculated = sum(evaluation["key_points_scores"].values())
            assert evaluation["total_score"] == total_score_calculated, \
                f"'total_score' in {model_name} is {evaluation['total_score']}, but calculated score is {total_score_calculated}"

        check_evaluation(response["caption_evaluation"], "caption_evaluation")
        return True
    except AssertionError as e:
        print(f"Validation failed: {e}")
        return False


def gpt_evaluate(prompt, api_key, model, max_retries=5):
    """Evaluate caption using GPT API"""
    client = OpenAI(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            response = completion.choices[0].message.content.strip()
            
            try:
                response_data = json.loads(response)
                if validate_response_format(response_data):
                    print(f'API call successful, normal return (attempt {attempt+1}/{max_retries})')
                    return response
                else:
                    print(f'Invalid response format, retrying (attempt {attempt+1}/{max_retries})')
            except json.JSONDecodeError:
                print(f'Response is not valid JSON format, retrying (attempt {attempt+1}/{max_retries})')
            except Exception as e:
                print(f'Error validating response: {e}, retrying (attempt {attempt+1}/{max_retries})')
            
            time.sleep(2)
        except Exception as e:
            print(f"API call error: {e}, retrying (attempt {attempt+1}/{max_retries})")
            time.sleep(2)
    
    print('API call failed, unable to get normal response')
    return '{"caption_evaluation": {"key_points_scores": {"error": 0}, "total_score": 0, "score_reasons": {"error": "API error after multiple attempts"}}'

def load_test_data(test_data_path):
    """Load test data with keypoints"""
    test_data_ids = set()
    test_data_dict = {}
    with open(test_data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            test_data_ids.add(data['id'])
            test_data_dict[data['id']] = data
    return test_data_ids, test_data_dict

def process_single_evaluation(data_item, test_data_dict, args, prompt_template):
    """Process evaluation for a single audio sample"""
    if data_item['id'] not in test_data_dict:
        return None
    
    test_data = test_data_dict[data_item['id']]
    answer = data_item['model_response_content']
    
    # Extract key points
    key_points = test_data.get('key_point', [])
    key_points_str = "\n".join(key_points)
    
    modified_prompt = prompt_template.replace("{key_points}", key_points_str).replace("{answer}", answer)
    response_gpt = gpt_evaluate(modified_prompt, args.api_key, args.model, args.max_retries)
    print(f'response:{response_gpt}')
    
    try:
        response_gpt_dict = json.loads(response_gpt)
        caption_eval = response_gpt_dict.get("caption_evaluation", {})
        
        result = {
            "id": data_item['id'],
            "restriction": data_item['restriction'],
            "audio": data_item['audio'],
            "question": test_data['conversations'][0]['value'] if 'conversations' in test_data else "",
            "reference": test_data['conversations'][1]['value'] if 'conversations' in test_data and len(test_data['conversations']) > 1 else "",
            "answer": answer,
            "requirements": key_points,
            "model_evaluation": caption_eval,
        }
        
        return result
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Original response: {response_gpt}")
        return None
    except Exception as e:
        print(f"Error processing audio {data_item['id']}: {e}")
        return None

def process_batch_evaluation(batch_data, test_data_dict, args, prompt_template, output_file):
    """Process evaluation for a batch of audio samples"""
    results = []
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_data = {
            executor.submit(process_single_evaluation, data_item, test_data_dict, args, prompt_template): data_item
            for data_item in batch_data
        }
        
        for future in as_completed(future_to_data):
            data_item = future_to_data[future]
            try:
                result = future.result()
                if result:
                    output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    output_file.flush()
                    results.append(result)
            except Exception as e:
                print(f"Error processing audio {data_item['id']}: {e}")
    
    return results

def count_total_words(mixed_str):
    """Count total words in text (supports mixed Chinese and English)"""
    en_words = re.findall(r"\b[a-zA-Z]+(?:['-][a-zA-Z]+)*\b", mixed_str)
    zh_chars = re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", mixed_str)
    return len(en_words) + len(zh_chars)

def sentences_count(text):
    """Count the number of sentences in text"""
    sentences = re.split(r'[.!?]', text)
    return sum(1 for s in sentences if s.strip())

def check_length_requirement(data, limit, caption, model_type):
    """Check if caption meets length limit requirements"""
    if "sentence" in limit:
        model_sentence_length = sentences_count(caption)
        
        sentence_requirements = {
            "The generated caption's length needs to be one sentence.": 1,
            "The generated caption's length needs to be exactly two sentences.": 2,
            "The generated caption's length cannot exceed two sentences.": (0, 2),
            "The generated caption's length needs to be exactly three sentences.": 3,
            "The generated caption's length cannot exceed three sentences.": (0, 3),
            "The generated caption's length needs to be exactly five sentences.": 5
        }
        
        if limit in sentence_requirements:
            requirement = sentence_requirements[limit]
            if isinstance(requirement, tuple):
                return model_sentence_length <= requirement[1]
            else:
                return model_sentence_length == requirement
            
    elif "word" in limit:
        words_model = count_total_words(caption)
        
        word_requirements = {
            "The generated caption's length needs to be no more than 10 words.": (0, 10),
            "The generated caption's length needs to be exactly 10 words.": 10,
            "The generated caption's length needs to be 10 to 20 words.": (10, 20),
            "The generated caption's length needs to be exactly 20 words.": 20,
            "The generated caption's length needs to be 20 to 30 words.": (20, 30),
            "The generated caption's length needs to be exactly 50 words.": 50,
            "The generated caption's length needs to be exactly 60 words.": 60,
            "The generated caption's length needs to be 30 to 120 words.": (30, 120)
        }
        
        if limit in word_requirements:
            requirement = word_requirements[limit]
            if isinstance(requirement, tuple):
                return requirement[0] <= words_model <= requirement[1]
            else:
                return words_model == requirement
    
    print(f'Warning: Unrecognized restriction condition: {limit}')
    return True

def calculate_scores_and_save(output_path):
    """Calculate and save score statistics"""
    total_score = 0
    max_score = 0
    total_entries = 0
    invalid_data_count = 0
    length_penalty_count = 0
    
    dimension_scores = {}
    
    with open(output_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                total_entries += 1

                if 'model_evaluation' not in data:
                    continue
                
                if 'total_score' not in data['model_evaluation']:
                    continue
                
                if 'answer' in data and 'None' in str(data['answer']):
                    continue

                score = data['model_evaluation']['total_score']
                requirements = data.get('requirements', [])
                
                requirements_count = len(requirements)
                if requirements_count == 0:
                    invalid_data_count += 1
                    continue
                
                max_score += requirements_count
                total_score += score

                # Check length restriction
                if 'restriction' in data and len(data['restriction']) > 1:
                    len_limit = data['restriction'][1]
                    if 'answer' in data:
                        answer = data['answer']
                        if not check_length_requirement(data, len_limit, answer, 'model'):
                            score = max(score - 1, 0)
                            length_penalty_count += 1
                
                # Dimension statistics
                if 'restriction' in data:
                    dimension = data['restriction'][0]
                    
                    if dimension not in dimension_scores:
                        dimension_scores[dimension] = {
                            'score': 0,
                            'max_score': 0,
                            'count': 0,
                            'total_length': 0
                        }
                    
                    dimension_scores[dimension]['score'] += score
                    dimension_scores[dimension]['max_score'] += requirements_count
                    dimension_scores[dimension]['count'] += 1
                    
                    if 'answer' in data:
                        dimension_scores[dimension]['total_length'] += count_total_words(str(data['answer']))
                                
            except json.JSONDecodeError:
                continue

    # Calculate percentage
    percentage = (total_score / max(max_score, 1)) * 100
    valid_entries = total_entries - invalid_data_count

    # Save summary
    summary_path = os.path.join(os.path.dirname(output_path), "audio_summary_with_length_penalty.txt")
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write("===== Audio Caption Evaluation Summary =====\n")
        summary_file.write(f"Total evaluated samples: {total_entries}\n")
        summary_file.write(f"Valid evaluated samples: {valid_entries}\n")
        summary_file.write(f"Invalid data count: {invalid_data_count}\n\n")
        
        summary_file.write("===== Length Penalty Statistics =====\n")
        summary_file.write(f"Applied length penalty count: {length_penalty_count} ({(length_penalty_count/valid_entries*100):.2f}%)\n\n")
        
        summary_file.write("===== Final Statistics =====\n")
        summary_file.write(f"Model score: {total_score}/{max_score} ({percentage:.2f}%)\n\n")
        
        summary_file.write("\n===== Evaluation Results by Dimension =====\n")
        for dimension, scores in dimension_scores.items():
            dim_percentage = (scores['score'] / max(scores['max_score'], 1)) * 100
            dim_count = scores['count']
            
            avg_score = scores['score'] / dim_count
            avg_length = scores['total_length'] / dim_count
            
            density = avg_score / avg_length if avg_length > 0 else 0
            
            summary_file.write(f"\nDimension: {dimension} ({dim_count} samples)\n")
            summary_file.write(f"  Score: {scores['score']}/{scores['max_score']} ({dim_percentage:.2f}%)\n")
            summary_file.write(f"  Average length: {avg_length:.2f} words\n")
            summary_file.write(f"  Density: {density * 100:.5f} (score/word count)\n")
        
    print(f"Evaluation summary saved to: {summary_path}")

def main(args):
    """Main processing function"""
    if args.calculate_only:
        print(f"Calculate score statistics only mode, using evaluation results: {args.output_path}")
        calculate_scores_and_save(args.output_path)
        return
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load test data with keypoints
    test_data_ids, test_data_dict = load_test_data(args.test_data_path)
    
    # Read prompt template
    with open(args.content_template_path, 'r', encoding='utf-8') as file:
        prompt_template = file.read()
    
    # Process data in batches
    with open(args.output_path, 'w', encoding='utf-8') as output_file:
        print(f'Opened output file: {args.output_path}')
        
        with open(args.data_path, 'r', encoding='utf-8') as file:
            print(f'Opened model result file: {args.data_path}')

            lines = file.readlines()
            batch_data = []
            
            for i, line in enumerate(tqdm(lines, desc='Processing audio samples')):
                data = json.loads(line)
                batch_data.append(data)
                
                if len(batch_data) >= args.batch_size or i == len(lines) - 1:
                    print(f"Processing batch: {len(batch_data)} samples")
                    process_batch_evaluation(batch_data, test_data_dict, args, prompt_template, output_file)
                    batch_data = []
    
    print('Completed batch evaluation, starting score calculation.')
    calculate_scores_and_save(args.output_path)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)