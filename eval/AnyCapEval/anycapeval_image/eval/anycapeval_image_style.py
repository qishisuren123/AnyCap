import json
import base64
import time
import requests
from tqdm import tqdm
import re
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Image description style evaluation tool')
    
    # API configuration parameters
    parser.add_argument('--base_url', type=str, default="https://boyuerichdata.chatgptten.com/v1/chat/completions",
                       help='API base URL')
    parser.add_argument('--api_key', type=str, default="your/api/key",
                       help='API key')
    parser.add_argument('--model', type=str, default="gpt-4o-2024-08-06",
                       help='Model name to use')
    
    # Data path parameters
    parser.add_argument('--data_path', type=str, default='/path/to/style/model_results.jsonl',
                       help='Original model results file path')
    parser.add_argument('--style_template_path', type=str, default='/path/to/instruction_style.txt',
                       help='Style evaluation prompt template file path')
    parser.add_argument('--image_dir', type=str, default='path/to/test/image/directory',
                       help='Directory containing image files')
    parser.add_argument('--output_path', type=str, default='/path/to/instruction_style.txt',
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
                       help='Batch size - number of images to process simultaneously')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Number of worker threads in thread pool')
    
    return parser.parse_args()

def create_client(args):
    """Create API client"""
    return {
        'base_url': args.base_url,
        'api_key': args.api_key,
        'headers': {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}"  
        }
    }

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def gpt_evaluate_style(modified_prompt, base64_image, client, model, max_retries=5):
    """Evaluate image caption style"""
    for attempt in range(max_retries):
        try:
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": modified_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 2000
            }
            
            response = requests.post(client['base_url'], headers=client['headers'], json=data)
            response_gpt = response.json()['choices'][0]['message']['content']

            try:
                response_data = json.loads(response_gpt)
                if isinstance(response_data, dict) and "score" in response_data and isinstance(response_data["score"], int) and "reason" in response_data:
                    print(f'API call successful (attempt {attempt+1})')
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

def sentences_count(text):
    """Count number of sentences in text"""
    sentences = re.split(r'[.!?]', text)
    return sum(1 for s in sentences if s.strip())

def count_total_words(mixed_str):
    """Count total words in text (supports mixed Chinese/English)"""
    en_words = re.findall(r"\b[a-zA-Z]+(?:['-][a-zA-Z]+)*\b", mixed_str)
    zh_chars = re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", mixed_str)
    return len(en_words) + len(zh_chars)

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

def process_image_pair(data_pair, prompt_template, image_dir, client, args):
    """Process single image evaluation"""
    data = data_pair.copy()
    try:
        data['source'] = 'data1'
        
        image_path = f"{image_dir}/{data['image']}"
        ref_answer = data['conversations'][1]['value'] if 'conversations' in data and len(data['conversations']) > 1 else ""
        answer = data.get('model_response_style', "")
        
        print(f"Processing image: {image_path}")
        base64_image = encode_image(image_path)

        restriction = data['restriction'][0]
        modified_prompt = prompt_template.replace("{caption_type}", restriction).replace("{output}", answer).replace("{reference}", ref_answer)
        
        response_gpt = gpt_evaluate_style(modified_prompt, base64_image, client, args.model, args.max_retries)
        try:
            response_data = json.loads(response_gpt)
            # Apply length penalty if needed
            if not check_length_requirement(data, None, answer, 'style'):
                response_data['score'] = min(response_data.get('score', 0), 1)
                response_data['length_penalty_applied'] = True
            data.update(response_data)
        except json.JSONDecodeError:
            print(f"Could not parse evaluation result: {response_gpt}")
            data.update({"score": 0, "reason": "Failed to parse evaluation result"})
        
        return data
    except Exception as e:
        print(f"Error evaluating image pair: {e}")
        return None

def compute_and_save_summary(args):
    """Compute and save evaluation summary"""
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

    # Save summary
    summary_path = os.path.join(os.path.dirname(args.output_path), "style_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write("===== Image Description Style Evaluation Summary =====\n")
        
        summary_file.write("\n----- Overall Evaluation Results -----\n")
        summary_file.write(f"Model average style score: {average_score:.2f}/4\n")
        
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
    """Process evaluation in batches"""
    client = create_client(args)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.style_template_path, 'r', encoding='utf-8') as file:
        prompt_template = file.read()
    
    with open(args.data_path, 'r', encoding='utf-8') as file:
        data_pairs = [json.loads(line) for line in file]
    
    mode = 'a' if args.append_mode else 'w'
    with open(args.output_path, mode, encoding='utf-8') as output_file:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for i in tqdm(range(0, len(data_pairs), args.batch_size), desc='Processing image batches'):
                batch = data_pairs[i:i+args.batch_size]
                
                futures = {
                    executor.submit(process_image_pair, pair, prompt_template, args.image_dir, client, args): idx
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
    """Main function"""
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