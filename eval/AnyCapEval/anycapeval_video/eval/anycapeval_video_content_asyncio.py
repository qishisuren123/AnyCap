import json
import base64
import requests
import time
import asyncio
import aiohttp
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
import openai
import re
import os
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import io
import cv2
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

print(f"OpenAI Version: {openai.__version__}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Video Caption Quality Evaluation Tool')
    
    # API configuration parameters
    parser.add_argument('--base_url', type=str, default="",
                        help='OpenAI API base URL')
    parser.add_argument('--api_key', type=str, default="",
                        help='OpenAI API key')
    parser.add_argument('--model', type=str, default="gpt-4o-mini",
                        help='Model name to use')
    
    # Data path parameters - changeable
    parser.add_argument('--data_path', type=str, 
                        default='',
                        help='Original model results file path')
    parser.add_argument('--output_path', type=str, 
                        default='/mnt/petrelfs/renyiming/ly_workspace/submit_code/anycapeval_video/output/code_test/eval_content.jsonl',
                        help='Evaluation result output file path')
    # Data path parameters - fixed
    parser.add_argument('--content_data_path', type=str, 
                        default='',
                        help='Predefined evaluation points data file path')
    parser.add_argument('--content_template_path', type=str, 
                        default='',
                        help='Prompt template file path')
    parser.add_argument('--video_dir', type=str, 
                        default='',
                        help='Video files directory')
    
    # Other parameters
    parser.add_argument('--max_retries', type=int, default=5,
                        help='Maximum retry count for API calls')
    parser.add_argument('--num_frames', type=int, default=4,
                        help='Number of frames to extract from each video')
    parser.add_argument('--calculate_only', action='store_true',
                        help='Only calculate score statistics from existing evaluation results')
    # Batch processing parameters
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size for simultaneous video processing')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Number of worker threads in thread pool')
    
    return parser.parse_args()

def create_client(base_url, api_key):
    return OpenAI(
        base_url=base_url,
        api_key=api_key
    )

def create_async_client(base_url, api_key):
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )

def validate_response_format(response):
    try:
        assert "caption_evaluation" in response, "Missing 'caption_evaluation'"

        def check_evaluation(evaluation, model_name):
            assert "key_points_scores" in evaluation, f"Missing 'key_points_scores' in {model_name}"
            assert "total_score" in evaluation, f"Missing 'total_score' in {model_name}"
            assert "score_reasons" in evaluation, f"Missing 'score_reasons' in {model_name}"

            # Check key_points_scores structure
            assert isinstance(evaluation["key_points_scores"], dict), f"'key_points_scores' in {model_name} must be a dict"
            for key_point, score in evaluation["key_points_scores"].items():
                assert isinstance(key_point, str), f"Key point '{key_point}' in {model_name} must be a string"
                assert isinstance(score, int), f"Score for '{key_point}' in {model_name} must be an integer"

            # Check total_score structure
            assert isinstance(evaluation["total_score"], int), f"'total_score' in {model_name} must be an integer"

            # Check score_reasons structure
            assert isinstance(evaluation["score_reasons"], dict), f"'score_reasons' in {model_name} must be a dict"
            for key_point, reason in evaluation["score_reasons"].items():
                assert isinstance(key_point, str), f"Key point '{key_point}' in {model_name} must be a string"
                assert isinstance(reason, str), f"Reason for '{key_point}' in {model_name} must be a string"

            # Check if keys in key_points_scores and score_reasons match
            assert set(evaluation["key_points_scores"].keys()) == set(evaluation["score_reasons"].keys()), \
                f"Keys in 'key_points_scores' and 'score_reasons' do not match in {model_name}"

            # Check if total score is correct
            total_score_calculated = sum(evaluation["key_points_scores"].values())
            assert evaluation["total_score"] == total_score_calculated, \
                f"'total_score' in {model_name} is {evaluation['total_score']}, but calculated score is {total_score_calculated}"

        # Check evaluation for both models
        check_evaluation(response["caption_evaluation"], "caption_evaluation")

        return True
    except AssertionError as e:
        print(f"Validation failed: {e}")
        return False

def extract_frames_from_video(video_path, num_frames=4):
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        
        total_frames = len(vr)
        if total_frames == 0:
            raise ValueError("Video has 0 frames")
        
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            frame = vr[idx].asnumpy()
            pil_img = Image.fromarray(frame)
            frames.append(pil_img)
            
        return frames
    except Exception as e:
        print(f"Error extracting frames from video: {e}")
        return extract_frames_with_cv2(video_path, num_frames)

def extract_frames_with_cv2(video_path, num_frames=4):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError("Video has 0 frames")
        
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                frames.append(pil_img)
        
        cap.release()
        return frames
    except Exception as e:
        print(f"Error extracting video frames with OpenCV: {e}")
        return []

def encode_image_to_base64(pil_image):
    """Encode PIL image to base64 string"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

async def async_gpt_evaluate(prompt, client, model, max_retries=5):
    for attempt in range(max_retries):
        try:
            message_content = [{"type": "text", "text": prompt}]
            
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                max_tokens=4096,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content.strip()
            
            try:
                response_data = json.loads(response_text)
                if validate_response_format(response_data):
                    print(f'API call successful, normal result returned (attempt {attempt+1}/{max_retries})')
                    return response_text
                else:
                    print(f'Invalid result format, retrying (attempt {attempt+1}/{max_retries})')
            except json.JSONDecodeError:
                print(f'Result is not valid JSON format, retrying (attempt {attempt+1}/{max_retries})')
            except Exception as e:
                print(f'Error validating result: {e}, retrying (attempt {attempt+1}/{max_retries})')
            
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"API call error: {e}, retrying (attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(2)
    
    print('API call exception occurred, unable to get normal result')
    return '{"caption_evaluation": {"key_points_scores": {"error": 0}, "total_score": 0, "score_reasons": {"error": "API error after multiple attempts"}}}'


async def process_batch_evaluation(batch_data, client, args, prompt_template, output_file, lock):
    tasks = []
    
    for data_item in batch_data:
        data_id = data_item['id']
        video_path = data_item['video_path']
        answer = data_item['answer']
        key_points_str = data_item['key_points_str']
        
        print(f"Processing video [{data_id}]: {video_path}")

        modified_prompt = prompt_template.replace("{key_points}", key_points_str).replace("{answer}", answer)
        
        task = asyncio.create_task(
            async_gpt_evaluate(modified_prompt, client, args.model, args.max_retries)
        )
        
        tasks.append((data_item, task))
    
    for data_item, task in tasks:
        try:
            response_gpt = await task
            
            try:
                response_gpt_dict = json.loads(response_gpt)
                
                model_evaluation = response_gpt_dict["caption_evaluation"]
                
                result = {
                    "id": data_item['id'],
                    "restriction": data_item['restriction'],
                    "video": data_item['video_name'],
                    "question": data_item['question'],
                    "reference": data_item['ref_answer'],
                    "answer": data_item['answer'],
                    "requirements": data_item['requirements'],
                    "model_evaluation": model_evaluation,
                }
                
                async with lock:
                    output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    output_file.flush()
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Original response: {response_gpt}")
        except Exception as e:
            print(f"Error processing video {data_item['id']}: {e}")

def prepare_data_batch(data, req_data, video_dir):
    data_id = data['id']
    
    video_path = f"{video_dir}/{data['video']}"
    question = data['conversations'][0]['value']
    ref_answer = data['conversations'][1]['value'] if 'conversations' in data and len(data['conversations']) > 1 else ""
    
    answer = data.get('model_response_content', "")[0]
    
    key_points = req_data.get('key_point', [])
    key_points_str = "\n".join(key_points)
    
    return {
        'id': data_id,
        'restriction': req_data['restriction'],
        'video_name': data['video'],
        'video_path': video_path,
        'question': question,
        'ref_answer': ref_answer,
        'answer': answer,
        'requirements': key_points,
        'key_points_str': key_points_str
    }

async def process_async_evaluation(args):

    async_client = create_async_client(args.base_url, args.api_key)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.content_template_path, 'r', encoding='utf-8') as file:
        prompt_template = file.read()
    
    requirements_data = {}  
    try:
        with open(args.content_data_path, 'r', encoding='utf-8') as req_file:
            for line in req_file:
                req_item = json.loads(line.strip())
                requirements_data[req_item['id']] = req_item 
        print(f'Loaded predefined evaluation points: {len(requirements_data)} entries')
    except FileNotFoundError:
        print(f"Error: Cannot find predefined evaluation points file {args.content_data_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid format in predefined evaluation points file")
        return
    
    with open(args.output_path, 'w', encoding='utf-8') as output_file:
        print(f'Opened output file: {args.output_path}')
        
        with open(args.data_path, 'r', encoding='utf-8') as file:
            print(f'Opened model results file: {args.data_path}')

            lines = file.readlines()
            
            # Create mutex lock
            lock = asyncio.Lock()
            
            batch_data = []
            batch_indices = []
            
            for i, line in enumerate(lines):
                actual_index = i
                
                data = json.loads(line)
                data_id = data['id']

                if data_id not in requirements_data:
                    continue
                
                req_data = requirements_data[data_id]
                
                data_item = prepare_data_batch(data, req_data, args.video_dir)
                batch_data.append(data_item)
                batch_indices.append(actual_index)
                
                if len(batch_data) >= args.batch_size or actual_index == len(lines) - 1:
                    print(f"Processing batch: indices {batch_indices[0]} to {batch_indices[-1]} ({len(batch_data)} videos)")
                    
                    await process_batch_evaluation(batch_data, async_client, args, prompt_template, output_file, lock)
                    
                    batch_data = []
                    batch_indices = []
    
    print('Completed async batch evaluation, starting score calculation.')
    calculate_scores_and_save(args.output_path)


def calculate_scores_and_save(output_path):
    pre_total_score = 0
    pre_max_score = 0
    
    post_total_score = 0
    post_max_score = 0
    
    total_entries = 0
    invalid_data_count = 0
    length_penalty_count = 0
    
    pre_dimension_scores = {}
    post_dimension_scores = {}
    
    path_dir = os.path.dirname(output_path)

    
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

                pre_score = data['model_evaluation']['total_score']

                requirements = data.get('requirements', [])
                key_points = data.get('key_point', [])
                
                requirements_count = len(requirements) if requirements else len(key_points)
                if requirements_count == 0:
                    invalid_data_count += 1
                    print(f"Warning: Invalid data at row {total_entries}, both requirements and key_point are empty")
                    continue
                
                pre_max_score += requirements_count
                
                pre_total_score += pre_score

                post_score = pre_score
                
                post_max_score += requirements_count
                
                # Check if length penalty needs to be applied
                if 'restriction' in data and len(data['restriction']) > 1:
                    len_limit = data['restriction'][1]
                    
                    # Apply length penalty
                    if 'answer' in data:
                        answer = data['answer']
                        if not check_length_requirement(data, len_limit, answer, 'model'):
                            post_score = max(post_score - 1, 0)
                            data['model_evaluation']['total_score'] = post_score
                            length_penalty_count += 1
                
                post_total_score += post_score
                
                if 'restriction' in data:
                    dimension = data['restriction'][0]
                    
                    if dimension not in pre_dimension_scores:
                        pre_dimension_scores[dimension] = {
                            'score': 0,
                            'max_score': 0,
                            'count': 0,
                            'total_length': 0
                        }
                    
                    pre_dimension_scores[dimension]['score'] += pre_score
                    pre_dimension_scores[dimension]['max_score'] += requirements_count
                    pre_dimension_scores[dimension]['count'] += 1
                    
                    if 'answer' in data:
                        pre_dimension_scores[dimension]['total_length'] += count_total_words(str(data['answer']))
                    
                    if dimension not in post_dimension_scores:
                        post_dimension_scores[dimension] = {
                            'score': 0,
                            'max_score': 0,
                            'count': 0,
                            'total_length': 0
                        }
                    
                    # Update dimension statistics (after penalty)
                    post_dimension_scores[dimension]['score'] += post_score
                    post_dimension_scores[dimension]['max_score'] += requirements_count
                    post_dimension_scores[dimension]['count'] += 1
                    
                    if 'answer' in data:
                        post_dimension_scores[dimension]['total_length'] += count_total_words(str(data['answer']))
                                
            except json.JSONDecodeError:
                continue

    # Calculate valid data entries
    valid_entries = total_entries - invalid_data_count

    # Calculate percentages
    pre_percentage = (pre_total_score / max(pre_max_score, 1)) * 100
    post_percentage = (post_total_score / max(post_max_score, 1)) * 100

    # Write evaluation summary
    summary_path = os.path.join(os.path.dirname(output_path), "content_summary_with_length_penalty.txt")
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write("===== Video Caption Evaluation Summary (Using Predefined Evaluation Points) =====\n")
        summary_file.write(f"Total evaluated videos: {total_entries}\n")
        summary_file.write(f"Valid evaluated videos: {valid_entries}\n")
        summary_file.write(f"Invalid data count: {invalid_data_count}\n\n")
        
        summary_file.write("===== Length Penalty Statistics =====\n")
        summary_file.write(f"Applied length penalty count: {length_penalty_count} ({(length_penalty_count/valid_entries*100):.2f}%)\n\n")
        
        summary_file.write("===== Statistics After Penalty =====\n")
        summary_file.write(f"Model score: {post_total_score}/{post_max_score} ({post_percentage:.2f}%)\n\n")
        
        summary_file.write("\n===== Evaluation Results by Dimension (After Penalty) =====\n")
        for dimension, scores in post_dimension_scores.items():
            dim_percentage = (scores['score'] / max(scores['max_score'], 1)) * 100
            dim_count = scores['count']
            
            avg_score = scores['score'] / dim_count
            avg_length = scores['total_length'] / dim_count
            
            density = avg_score / avg_length if avg_length > 0 else 0
            
            summary_file.write(f"\nDimension: {dimension} ({dim_count} videos)\n")
            summary_file.write(f"  Score: {scores['score']}/{scores['max_score']} ({dim_percentage:.2f}%)\n")
            summary_file.write(f"  Average length: {avg_length:.2f} words\n")
            summary_file.write(f"  Density: {density * 100:.5f} (score/word count)\n")
        
    print(f"Evaluation summary saved to: {summary_path}")

def sentences_count(text):
    """Count the number of sentences in text"""
    sentences = re.split(r'[.!?]', text)
    return sum(1 for s in sentences if s.strip())

def count_total_words(mixed_str):
    """Count total words in text (supports mixed Chinese and English)"""
    en_words = re.findall(r"\b[a-zA-Z]+(?:['-][a-zA-Z]+)*\b", mixed_str)
    zh_chars = re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", mixed_str)
    return len(en_words) + len(zh_chars)

def check_length_requirement(data, limit, caption, model_type):
    """Check if caption meets length limitation requirements"""
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
            print(f"Model {model_type}, failed restriction: {limit}, {data['restriction'][0]}")
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
            print(f"Model {model_type}, failed restriction: {limit}, Restriction: {data['restriction'][0]}, actual word count: {words_model} words. ID: {data.get('id', 'unknown')}")
            return False
    
    print(f'Warning: Unrecognized restriction condition: {limit}')
    return True


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.calculate_only:
        print(f"Score calculation only mode, using evaluation results: {args.output_path}")
        calculate_scores_and_save(args.output_path)
    else:
        if sys.version_info >= (3, 7) and 'AsyncOpenAI' in dir(openai):
            print("Using async batch processing mode for evaluation...")
            asyncio.run(process_async_evaluation(args))