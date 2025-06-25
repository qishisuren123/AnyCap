# import json
# import base64
# import time
# import argparse
# import re
# from tqdm import tqdm
# from collections import defaultdict
# from openai import OpenAI

# client = OpenAI(
#     api_key='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',
# )

# def gpt_reference_nocontent(modified_prompt, base64_audio):
#     attempt = 5
#     while attempt > 0:
#         response = client.chat.completions.create(
#             model="gpt-4o-audio-preview",
#             messages=[{
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": modified_prompt},
#                     {
#                         "type": "input_audio",
#                         "input_audio": {
#                             "data": base64_audio,
#                             "format": "wav"
#                         }
#                     }
#                 ]
#             }]
#         )
#         response_gpt = response.choices[0].message.content.strip()
#         try:
#             response_data = json.loads(response_gpt)
#             if isinstance(response_data, dict) and "score" in response_data and isinstance(response_data["score"], int):
#                 return response_data
#         except Exception:
#             pass
#         attempt -= 1
#         time.sleep(1)
#     return {"score": "error", "reason": "error"}

# def encode_audio(audio_path):
#     with open(audio_path, "rb") as f:
#         return base64.b64encode(f.read()).decode('utf-8')

# def sentences_count(text):
#     return sum(1 for s in re.split(r'[.!?]', text) if s.strip())

# def count_total_words(mixed_str):
#     en_words = re.findall(r"\b[a-zA-Z]+(?:['-][a-zA-Z]+)*\b", mixed_str)
#     zh_chars = re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", mixed_str)
#     return len(en_words) + len(zh_chars)

# def sentence_modify(limit, caption):
#     model_sentence_length = sentences_count(caption)
#     expected_conditions = {
#         "one": 1, "two": 2, "three": 3, "five": 5,
#     }
#     for keyword, expected in expected_conditions.items():
#         if f"exactly {keyword}" in limit:
#             return model_sentence_length == expected
#         elif f"cannot exceed {keyword}" in limit:
#             return model_sentence_length <= expected
#     return True

# def word_modify(limit, caption):
#     words_model = count_total_words(caption)
#     if "no more than 10 words" in limit:
#         return words_model <= 10
#     if "exactly 10 words" in limit:
#         return words_model == 10
#     if "10 to 20 words" in limit:
#         return 10 <= words_model <= 20
#     if "exactly 20 words" in limit:
#         return words_model == 20
#     if "20 to 30 words" in limit:
#         return 20 <= words_model <= 30
#     if "exactly 50 words" in limit:
#         return words_model == 50
#     if "exactly 60 words" in limit:
#         return words_model == 60
#     if "30 to 120 words" in limit:
#         return 30 <= words_model <= 120
#     return True

# def apply_length_constraints(entry):
#     caption = entry['model_response_content']
#     length_limit = entry['restriction'][1]
#     if "sentence" in length_limit:
#         return sentence_modify(length_limit, caption)
#     elif "word" in length_limit:
#         return word_modify(length_limit, caption)
#     return True

# def print_final_scores(output_file):
#     scores_by_type = defaultdict(list)
#     with open(output_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data = json.loads(line)
#                 score = data.get("score")
#                 if not isinstance(score, (int, float)):
#                     continue
#                 key = data.get("restriction", [None])[0]
#                 if key:
#                     scores_by_type[key].append(score)
#             except Exception:
#                 continue

#     print("\n📊 最终分数统计（按 Caption 类型）:")
#     for caption_type in sorted(scores_by_type.keys()):
#         scores = scores_by_type[caption_type]
#         if scores:
#             avg = sum(scores) / len(scores)
#             print(f"  {caption_type}: {avg:.2f}（共 {len(scores)} 条）")
#         else:
#             print(f"  {caption_type}: 无有效分数")

# def main(data_path, prompt_path, audio_dir, final_output_path):
#     print("==> Step 1: Run GPT evaluator")
#     entries = []
#     with open(prompt_path, 'r', encoding='utf-8') as f:
#         prompt_template = f.read()

#     with open(data_path, 'r', encoding='utf-8') as f_in:
#         for line in tqdm(f_in, desc="Evaluating"):
#             try:
#                 data = json.loads(line)
#                 ref_answer = data["conversations"][1]["value"]
#                 model_output = data["model_response_content"]
#                 restriction = data['restriction'][0]
#                 audio_path = f"{audio_dir}/{data['audio']}"
#                 base64_audio = encode_audio(audio_path)

#                 prompt = prompt_template.replace("{caption_type}", restriction).replace("{output}", model_output).replace("{reference}", ref_answer)
#                 response = gpt_reference_nocontent(prompt, base64_audio)
#                 data.update(response)
#                 entries.append(data)
#             except Exception:
#                 continue

#     print("==> Step 2: Apply length limit post-processing")
#     modified_count = 0
#     with open(final_output_path, 'w', encoding='utf-8') as f_out:
#         for entry in entries:
#             if entry.get('score') == 'error':
#                 continue
#             if entry.get('model_response_nocontent', '') == 'N':
#                 continue
#             try:
#                 original_score = int(entry['score'])
#                 if not apply_length_constraints(entry):
#                     entry['score'] = min(original_score, 1)
#                     modified_count += 1
#             except Exception:
#                 continue
#             f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

#     print(f"\n✅ 最终输出已保存至：{final_output_path}")
#     print(f"⚠️ 被修改分数的数量：{modified_count}")
#     print_final_scores(final_output_path)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data", type=str, default='', help="模型输出数据路径")
#     parser.add_argument("--prompt", type=str, default='', help="prompt 模板路径")
#     parser.add_argument("--audio_dir", type=str, default='', help="音频文件夹路径")
#     parser.add_argument("--final_output", type=str, default='', help="最终输出 JSONL 文件路径")
#     args = parser.parse_args()

#     main(args.data, args.prompt, args.audio_dir, args.final_output)

# import json
# import base64
# import time
# import argparse
# import re
# from tqdm import tqdm
# from collections import defaultdict
# from openai import OpenAI

# client = OpenAI(
#     api_key='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',
# )

# def gpt_reference_nocontent(modified_prompt, base64_audio):
#     attempt = 5
#     while attempt > 0:
#         response = client.chat.completions.create(
#             model="gpt-4o-audio-preview",
#             messages=[{
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": modified_prompt},
#                     {
#                         "type": "input_audio",
#                         "input_audio": {
#                             "data": base64_audio,
#                             "format": "wav"
#                         }
#                     }
#                 ]
#             }]
#         )
#         response_gpt = response.choices[0].message.content.strip()
#         try:
#             response_data = json.loads(response_gpt)
#             if isinstance(response_data, dict) and "score" in response_data and isinstance(response_data["score"], int):
#                 return response_data
#         except Exception:
#             pass
#         attempt -= 1
#         time.sleep(1)
#     return {"score": "error", "reason": "error"}

# def encode_audio(audio_path):
#     with open(audio_path, "rb") as f:
#         return base64.b64encode(f.read()).decode('utf-8')

# def sentences_count(text):
#     return sum(1 for s in re.split(r'[.!?]', text) if s.strip())

# def count_total_words(mixed_str):
#     en_words = re.findall(r"\b[a-zA-Z]+(?:['-][a-zA-Z]+)*\b", mixed_str)
#     zh_chars = re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", mixed_str)
#     return len(en_words) + len(zh_chars)

# def sentence_modify(limit, caption):
#     model_sentence_length = sentences_count(caption)
#     expected_conditions = {
#         "one": 1, "two": 2, "three": 3, "five": 5,
#     }
#     for keyword, expected in expected_conditions.items():
#         if f"exactly {keyword}" in limit:
#             return model_sentence_length == expected
#         elif f"cannot exceed {keyword}" in limit:
#             return model_sentence_length <= expected
#     return True

# def word_modify(limit, caption):
#     words_model = count_total_words(caption)
#     if "no more than 10 words" in limit:
#         return words_model <= 10
#     if "exactly 10 words" in limit:
#         return words_model == 10
#     if "10 to 20 words" in limit:
#         return 10 <= words_model <= 20
#     if "exactly 20 words" in limit:
#         return words_model == 20
#     if "20 to 30 words" in limit:
#         return 20 <= words_model <= 30
#     if "exactly 50 words" in limit:
#         return words_model == 50
#     if "exactly 60 words" in limit:
#         return words_model == 60
#     if "30 to 120 words" in limit:
#         return 30 <= words_model <= 120
#     return True

# def apply_length_constraints(entry):
#     caption = entry['model_response_content']
#     length_limit = entry['restriction'][1]
#     if "sentence" in length_limit:
#         return sentence_modify(length_limit, caption)
#     elif "word" in length_limit:
#         return word_modify(length_limit, caption)
#     return True

# def print_final_scores(output_file):
#     scores_by_type = defaultdict(list)
#     with open(output_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data = json.loads(line)
#                 score = data.get("score")
#                 if not isinstance(score, (int, float)):
#                     continue
#                 key = data.get("restriction", [None])[0]
#                 if key:
#                     scores_by_type[key].append(score)
#             except Exception:
#                 continue

#     summary_path = output_file.replace('.jsonl', '_summary.txt')
#     with open(summary_path, 'w', encoding='utf-8') as summary_file:
#         print("\n📊 最终分数统计（按 Caption 类型）:")
#         summary_file.write(f"📊 最终分数统计（按 Caption 类型）:\n")

#         for caption_type in sorted(scores_by_type.keys()):
#             scores = scores_by_type[caption_type]
#             if scores:
#                 avg = sum(scores) / len(scores)
#                 line = f"  {caption_type}: {avg:.2f}（共 {len(scores)} 条）"
#             else:
#                 line = f"  {caption_type}: 无有效分数"
#             print(line)
#             summary_file.write(line + "\n")

#     print(f"\n📄 分类平均分结果已保存至：{summary_path}")

# def main(data_path, prompt_path, audio_dir, final_output_path):
#     print("==> Step 1: Run GPT evaluator")
#     entries = []
#     with open(prompt_path, 'r', encoding='utf-8') as f:
#         prompt_template = f.read()

#     with open(data_path, 'r', encoding='utf-8') as f_in:
#         for line in tqdm(f_in, desc="Evaluating"):
#             try:
#                 data = json.loads(line)
#                 ref_answer = data["conversations"][1]["value"]
#                 model_output = data["model_response_content"]
#                 restriction = data['restriction'][0]
#                 audio_path = f"{audio_dir}/{data['audio']}"
#                 base64_audio = encode_audio(audio_path)

#                 prompt = prompt_template.replace("{caption_type}", restriction).replace("{output}", model_output).replace("{reference}", ref_answer)
#                 response = gpt_reference_nocontent(prompt, base64_audio)
#                 data.update(response)
#                 entries.append(data)
#             except Exception:
#                 continue

#     print("==> Step 2: Apply length limit post-processing")
#     modified_count = 0
#     with open(final_output_path, 'w', encoding='utf-8') as f_out:
#         for entry in entries:
#             if entry.get('score') == 'error':
#                 continue
#             if entry.get('model_response_nocontent', '') == 'N':
#                 continue
#             try:
#                 original_score = int(entry['score'])
#                 if not apply_length_constraints(entry):
#                     entry['score'] = min(original_score, 1)
#                     modified_count += 1
#             except Exception:
#                 continue
#             f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

#     print(f"\n✅ 最终输出已保存至：{final_output_path}")
#     print(f"⚠️ 被修改分数的数量：{modified_count}")
#     print_final_scores(final_output_path)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data", type=str, default='', help="模型输出数据路径")
#     parser.add_argument("--prompt", type=str, default='', help="prompt 模板路径")
#     parser.add_argument("--audio_dir", type=str, default='', help="音频文件夹路径")
#     parser.add_argument("--final_output", type=str, default='', help="最终输出 JSONL 文件路径")
#     args = parser.parse_args()

#     main(args.data, args.prompt, args.audio_dir, args.final_output)


import json
import base64
import time
import argparse
import re
import os
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

def parse_arguments():
    """Parse command line arguments (aligned with image version)"""
    parser = argparse.ArgumentParser(description='Audio caption quality evaluation tool')
    
    # API configuration parameters
    parser.add_argument('--api_key', type=str, default='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',
                       help='API key')
    parser.add_argument('--model', type=str, default="gpt-4o-audio-preview",
                       help='Model name to use')
    
    # Data path parameters
    parser.add_argument('--data_path', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/eval/input/style_input/test.jsonl',
                       help='Original model results file path')
    parser.add_argument('--prompt_template_path', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/instruction_style.txt',
                       help='Evaluation prompt template file path')
    parser.add_argument('--audio_dir', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/test_audio_data',
                       help='Audio files directory')
    parser.add_argument('--output_path', type=str, default='/mnt/petrelfs/renyiming/gm_workspace/audio_submit_code/anycapeval_audio/eval/output/test_style.jsonl',
                       help='Evaluation results output file path')
    
    # Processing parameters
    parser.add_argument('--max_retries', type=int, default=5,
                       help='Maximum API call retries')
    parser.add_argument('--append_mode', action='store_true',
                       help='Use append mode for output file writing')
    parser.add_argument('--calculate_only', action='store_true',
                       help='Only calculate score statistics from existing evaluation results')
    
    # Batch processing parameters
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Batch size - number of audios to process simultaneously')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Number of worker threads in thread pool')
    
    return parser.parse_args()

def create_client(args):
    """Create API client (aligned with image version)"""
    return OpenAI(api_key=args.api_key)

def encode_audio(audio_path):
    """Encode audio to base64 (renamed from encode_audio to match image version pattern)"""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def count_total_words(mixed_str):
    """Count total words in text (identical to image version)"""
    en_words = re.findall(r"\b[a-zA-Z]+(?:['-][a-zA-Z]+)*\b", mixed_str)
    zh_chars = re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", mixed_str)
    return len(en_words) + len(zh_chars)

def sentences_count(text):
    """Count sentences in text (kept from original audio version)"""
    return sum(1 for s in re.split(r'[.!?]', text)) if s.strip() else 0

def check_length_requirement(data, limit, caption, model_type):
    """Check if caption meets length requirements"""
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
                if model_sentence_length <= requirement[1]:
                    return True
            else:
                if model_sentence_length == requirement:
                    return True
            print(f"Model {model_type}, failed requirement: {limit},{data['restriction'][0]}")
            return False
            
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
                if requirement[0] <= words_model <= requirement[1]:
                    return True
            else:
                if words_model == requirement:
                    return True
            print(f"Model {model_type}, failed requirement: {limit}, Restriction {data['restriction'][0]}, actual word count {words_model} words. ID: {data.get('id', 'unknown')}")
            return False
    
    print(f'Note: Unrecognized requirement: {limit}')
    return True

def gpt_evaluate_style(modified_prompt, base64_audio, client, model, max_retries=5):
    """Evaluate audio caption content (aligned with image version naming)"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": modified_prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64_audio,
                                "format": "wav"
                            }
                        }
                    ]
                }]
            )
            response_gpt = response.choices[0].message.content.strip()
            
            try:
                response_data = json.loads(response_gpt)
                if isinstance(response_data, dict) and "score" in response_data and isinstance(response_data["score"], int):
                    return response_gpt
                else:
                    print(f'Invalid response format, retrying (attempt {attempt+1}/{max_retries})')
            except json.JSONDecodeError:
                print(f'Response not valid JSON, retrying (attempt {attempt+1}/{max_retries})')
            
            time.sleep(2)
            
        except Exception as e:
            print(f"API call error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
    
    print('API call failed after multiple attempts')
    return '{"score": 0, "reason": "API error occurred after multiple attempts"}'

def process_audio_pair(data_pair, prompt_template, audio_dir, client, args):
    """Process single audio evaluation (aligned with image version structure)"""
    data = data_pair.copy()
    try:
        data['source'] = 'data1'
        
        audio_path = f"{audio_dir}/{data['audio']}"
        ref_answer = data['conversations'][1]['value'] if 'conversations' in data and len(data['conversations']) > 1 else ""
        answer = data.get('model_response_style', "")
        
        print(f"Processing audio: {audio_path}")
        base64_audio = encode_audio(audio_path)

        restriction = data['restriction'][0]
        modified_prompt = prompt_template.replace("{caption_type}", restriction).replace("{output}", answer).replace("{reference}", ref_answer)
        
        response_gpt = gpt_evaluate_style(modified_prompt, base64_audio, client, args.model, args.max_retries)
        try:
            response_data = json.loads(response_gpt)
            # Apply length penalty if needed
            if not check_length_requirement(data, data['restriction'][1], answer, 'style'):
                response_data['score'] = min(response_data.get('score', 0), 1)
                response_data['length_penalty_applied'] = True
            data.update(response_data)
        except json.JSONDecodeError:
            print(f"Could not parse evaluation result: {response_gpt}")
            data.update({"score": 0, "reason": "Failed to parse evaluation result"})
        
        return data
    except Exception as e:
        print(f"Error evaluating audio pair: {e}")
        return None

def compute_and_save_summary(args):
    """Compute and save evaluation summary (aligned with image version)"""
    valid_scores = []
    dimension_data = {}  # Format: {dimension: {'scores': [], 'lengths': [], 'penalized_scores': []}}
    invalid_data_count = {'data1': 0, 'other': 0}
    length_penalty_count = 0
    total_valid = 0

    with open(args.output_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                score = data.get('score')
                source = data.get('source')
                reason = data.get('reason', '')
                
                if reason == "Failed to parse evaluation result":
                    invalid_data_count[source if source in invalid_data_count else 'other'] += 1
                    continue

                dimension = data.get('restriction', [None])[0]
                caption = data.get('model_response_style', "")
                
                if isinstance(score, (int, float)) and source == 'data1':
                    total_valid += 1
                    length = count_total_words(caption) if caption else 0
                    
                    # Record length penalty
                    if data.get('length_penalty_applied', False):
                        length_penalty_count += 1
                    
                    valid_scores.append(score)
                    
                    if dimension:
                        if dimension not in dimension_data:
                            dimension_data[dimension] = {
                                'scores': [],
                                'lengths': [],
                                'penalized_scores': []
                            }
                        dimension_data[dimension]['scores'].append(score)
                        dimension_data[dimension]['lengths'].append(length)
                        dimension_data[dimension]['penalized_scores'].append(score)
                        
            except json.JSONDecodeError:
                invalid_data_count['other'] += 1

    # Calculate statistics
    average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    dimension_stats = {}
    
    for dimension, data in dimension_data.items():
        if data['scores']:
            avg_score = sum(data['scores']) / len(data['scores'])
            avg_penalized_score = sum(data['penalized_scores']) / len(data['penalized_scores'])
            avg_length = sum(data['lengths']) / len(data['lengths'])
            density = avg_penalized_score / avg_length if avg_length > 0 else 0
            
            dimension_stats[dimension] = {
                'avg_score': avg_score,
                'avg_penalized_score': avg_penalized_score,
                'avg_length': avg_length,
                'density': density
            }
    
    print(dimension_stats)

    # Save summary
    summary_path = os.path.join(os.path.dirname(args.output_path), "style_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write("===== Audio Caption Content Evaluation Summary =====\n")
        
        summary_file.write("\n----- Overall Evaluation Results -----\n")
        summary_file.write(f"Model average content score: {average_score:.2f}/4\n")
        
        summary_file.write("\n----- Length Penalty Statistics -----\n")
        penalty_percent = (length_penalty_count / total_valid * 100) if total_valid > 0 else 0
        summary_file.write(f"Length penalty count: {length_penalty_count} ({penalty_percent:.2f}%)\n")
        
        summary_file.write("\n----- Evaluation Results by Dimension -----\n")
        for dimension, stats in dimension_stats.items():
            summary_file.write(f"\nDimension: {dimension}\n")
            summary_file.write(f"  Original average score: {stats['avg_score']:.2f}/4\n")
            summary_file.write(f"  Penalized average score: {stats['avg_penalized_score']:.2f}/4\n")
            summary_file.write(f"  Average length: {stats['avg_length']:.2f} words\n")
            summary_file.write(f"  Density: {stats['density']:.4f} (score/word count)\n")
        
        summary_file.write("\n----- Data Statistics -----\n")
        summary_file.write(f"Total evaluated: {total_valid + sum(invalid_data_count.values())}\n")
        summary_file.write(f"Valid data: {total_valid}\n")
        summary_file.write(f"Invalid data: {sum(invalid_data_count.values())}\n")
    
    print(f"Evaluation summary saved to: {summary_path}")

def process_evaluation_batch(args):
    """Process evaluation in batches (aligned with image version)"""
    client = create_client(args)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.prompt_template_path, 'r', encoding='utf-8') as file:
        prompt_template = file.read()
    
    with open(args.data_path, 'r', encoding='utf-8') as file:
        data_pairs = [json.loads(line) for line in file]
    
    mode = 'a' if args.append_mode else 'w'
    with open(args.output_path, mode, encoding='utf-8') as output_file:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for i in tqdm(range(0, len(data_pairs), args.batch_size), desc='Processing audio batches'):
                batch = data_pairs[i:i+args.batch_size]
                
                futures = {
                    executor.submit(process_audio_pair, pair, prompt_template, args.audio_dir, client, args): idx
                    for idx, pair in enumerate(batch, start=i)
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                            output_file.flush()
                    except Exception as e:
                        print(f"Error processing batch: {e}")

def main():
    """Main function (aligned with image version)"""
    args = parse_arguments()
    
    if args.calculate_only:
        print(f"Calculating scores from existing results: {args.output_path}")
        compute_and_save_summary(args)
    else:
        print(f"Starting batch evaluation (batch size: {args.batch_size}, workers: {args.max_workers})")
        process_evaluation_batch(args)
        compute_and_save_summary(args)

if __name__ == '__main__':
    main()