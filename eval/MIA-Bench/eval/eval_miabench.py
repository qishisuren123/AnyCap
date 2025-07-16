import os
import json
import pandas as pd
from tqdm import tqdm
import requests
import base64
import time
import argparse

base_url = "https://boyuerichdata.chatgptten.com/v1/chat/completions"
skey = ""
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {skey}"  
}


def process_rawscore(component_type, raw_score):
    first_sentence = raw_score.split('''.''')[0].split(''',''')
    score_dict = {}
    for i in range(len(first_sentence) - 1):
        score_ = first_sentence[i].split(''':''')[1][1:].split('''/''')
        score = int(score_[0])/int(score_[1])
        score_dict[component_type[i]] = score
    total_score_ = first_sentence[i+1].split(''':''')[1][1:].split('''/''')
    total_score = int(total_score_[0])/int(total_score_[1])
    score_dict['total_score'] = total_score
    return score_dict  

def get_score_dict(column_name, df):
    cat_score_dict = {}
    for i in range(len(df)):
        try:
            score_dict = process_rawscore(df['component_type'][i], df[column_name][i])
            for key, val in score_dict.items():
                if key not in cat_score_dict.keys():
                    cat_score_dict[key] = [val]
                else:
                    cat_score_dict[key].append(val)
        except:
            pass
    cat_score_dict_average = {}
    for key, val in cat_score_dict.items():
        cat_score_dict_average[key] = sum(val)/len(val)
    return cat_score_dict_average

    
def generate_prompt(d, response):
    instruction = d['instruction']
    weight = d['component_weight'] * 1
    d['num_of_component'] = len(d['components'])
    for i in range(len(weight)):
        weight[i] = str(weight[i])
    if d['num_of_component'] == 1:
        components = '''The first component is:' ''' + d['components'][0] + "'"  
        score = '''The first component is worth ''' + weight[0] + ' scores.'
    elif d['num_of_component'] == 2:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + "'" 
        score = '''The first and second component is each worth ''' + weight[0] + ' and ' + weight[1]+ ' scores.'
    elif d['num_of_component'] == 3:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + '''', and the third component is:' ''' + d['components'][2] + "'" 
        score = '''The first second, and third component is each worth ''' + weight[0] + ', ' + weight[1]+ ' and ' + weight[2] + ' scores.'
    elif d['num_of_component'] == 4:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + '''', and the third component is:' ''' + d['components'][2] +  '''', and the fourth component is:' ''' + d['components'][3] + "'" 
        score = '''The first second, third, and fourth component is each worth ''' + weight[0] + ', ' + weight[1]+ ', ' + weight[2] + ' and ' + weight[3] + ' scores.'
    elif d['num_of_component'] == 5:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + '''', and the third component is:' ''' + d['components'][2] +  '''', and the fourth component is:' ''' + d['components'][3] +  '''', and the fifth component is:' ''' + d['components'][4] + "'" 
        score = '''The first second, third, fourth and fifth component is each worth ''' + weight[0] + ', ' + weight[1]+ ', ' + weight[2] + ', ' + weight[3] + ' and ' + weight[4] + ' scores.'      
    return '''Here is an instruction for a multimodal LLM: ' ''' + instruction + ''' You need to grade if the response from the model follows each component of the instruction. ''' + components + ''' The response is:' '''  + response +  '''' You need to score the response and be strict. The total score ranges from 0 to 10, depending on if the response follows the instruction. ''' + score + ' List scores of each component, and the total score in one sentence in this format: score of component 1: x/2, score of component 2: y/8, total score: z/10. Then explain your reasons.'


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main(MIA_file, ans_file, output_txt_file):

    df = pd.read_json(MIA_file)
    answers = [json.loads(q) for q in open(ans_file, 'r')]

    df_intenrnvl2_5_8b = pd.DataFrame(answers)
    df_intenrnvl2_5_8b['score_raw'] = [_ for _ in range(len(df_intenrnvl2_5_8b))]

    results = []
    count = 0
    
    for i in tqdm(range(len(df_intenrnvl2_5_8b))):
        d = {}
        for col in df.columns:
            d[col] = df[col][i]
        response = df_intenrnvl2_5_8b['text'][i]
        image_path_original = df_intenrnvl2_5_8b['question_id'][i]
        image_name = os.path.basename(image_path_original)
        image_path = '/mnt/petrelfs/renyiming/lzq_workspace/MIA-Bench/image/' + image_name
        question =  generate_prompt(d, response)
        generated = False
        base64_image = encode_image(image_path)


        if df_intenrnvl2_5_8b['text'][i] != 'error':
            attempt = 3
            
            while attempt > 0 and generated == False:
                try:
                    data = {
                        "model":"gpt-4o-2024-08-06",
                        "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": question},
                                            {"type": "image_url",
                                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                            },
                                        ],
                                    }
                                ],
                                "max_tokens":4096
                    }
                    response = requests.post(base_url, headers=headers, json=data)

                    # 中转判断结果
                    if response.status_code == 200:
                        response_gpt = response.json()['choices'][0]['message']['content']
                        print("Response JSON:", response.json())
                        if "total score" not in response_gpt and "Total Score" not in response_gpt and "Total score" not in response_gpt:
                            print("Response does not contain 'total score'. Retrying...")
                            attempt -= 1  # 减少尝试次数
                            time.sleep(1)
                            continue  # 跳过后续代码，重新生成结果
                        df_intenrnvl2_5_8b['score_raw'][i] = response_gpt # 中转的结果
                        generated = True
                        count += 1
                    else:
                        print(f"Request failed with status code {response.status_code}")

                        print("Response:", response.text)
                        time.sleep(1)


                except Exception as e:
                    attempt -= 1
                    print("Error occurred:", e)
                    time.sleep(1)

        else:
            print('pass')
            pass
        
        # 将结果存入列表
        results.append({
            "question": question,
            "response": df_intenrnvl2_5_8b['score_raw'][i]
        })

    df['internvl2-5_8b'] = df_intenrnvl2_5_8b['score_raw']
    score = get_score_dict('internvl2-5_8b', df)

    #save results to the benchmark dataframe, for easier later comparison
    # 将结果保存为 TXT 文件
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"Question: {result['question']}\n")
            f.write(f"Response: {result['response']}\n\n")
        f.write(f'分数为：{score}')

    print(f"Results saved to {output_txt_file}")

    print('分数为：', score)

    print(f'有效数据(包含total score)数量为：{count}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate captions using GPT-4o")
    parser.add_argument("--MIA_file", type=str,
                        default="image_submit_code/MIA-Bench/instruction_benchmark_all.json",
                        help="MIA-Bench original file path")
    parser.add_argument("--ans_file", type=str,
                        default="/image_submit_code/MIA-Bench/output/inter_jsonl/InternVL2.5-8B_output.jsonl",
                        help="Generated captions file path")
    parser.add_argument("--output_txt_file", type=str,
                        default="/image_submit_code/MIA-Bench/output/result/InternVL2.5-8B_output.txt",
                        help="Output evaluation results file path")
    
    args = parser.parse_args()
    main(args.MIA_file, args.ans_file, args.output_txt_file)