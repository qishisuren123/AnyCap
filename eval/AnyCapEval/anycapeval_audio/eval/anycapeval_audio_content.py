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

DEFAULT_MAX_RETRIES = 5
DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_WORKERS = 4

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Audio caption quality assessment tool')
    
    # API configuration parameters
    parser.add_argument('--api_key', type=str, default='your/api/key',
                        help='OpenAI API key')
    parser.add_argument('--model', type=str, default="gpt-4o",
                        help='Model name to use')
    
    # Data path parameters
    parser.add_argument('--data_path', type=str, default='/path/to/content/model_results.jsonl',
                        help='Original model result file path')
    parser.add_argument('--output_path', type=str, default='/path/to/evaluation_results.jsonl',
                        help='Evaluation result output file path')
    parser.add_argument('--content_template_path', type=str, default='/path/to/instruction_content.txt',
                        help='Prompt template file path')
    parser.add_argument('--test_data_path', type=str, default='/path/to/anycapeval_audio_ref.jsonl',
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