from openai import OpenAI
import anthropic
from google import genai
import time
from tqdm import tqdm
import pandas as pd
from collections import Counter
import random
import numpy as np
import re
import json
import ast


def read_text(path): 
    with open(path, 'r', encoding='utf-8') as f: 
        content = f.read()
    return str(content)

def completion_gpt(model, messages, max_try=3, prt=False):
    client = OpenAI(
        api_key='your api key'
    )
    message = ''
    # try again when fail
    for i in range(max_try):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            message = response.choices[0].message.content
            if prt:
                print('{} response:'.format(model))
                print(message)
            break
        except Exception as e:
            print(e)
            time.sleep(0.5)
            continue
    return message

def apply_claude(model, messages, max_try=3, prt=False):
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key="your api key",
    )
    # try again when fail
    for i in range(max_try):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                messages=messages
            )
            message = response.content
            if prt:
                print('{} response:'.format(model))
                print(message)
            break
        except Exception as e:
            print(e)
            time.sleep(0.5)
            continue
    return message

def apply_gemini(model, content, max_try=3, prt=False):
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()
    for i in range(max_try):
        try:
            response = client.models.generate_content(
                model=model,
                contents=content
            )
            message = response.text
            if prt:
                print('{} response:'.format(model))
                print(message)
            break
        except Exception as e:
            print(e)
            time.sleep(0.5)
            continue
    return message


def safe_parse_json(text, idx=None):
    try:
        text = text.strip()
        # clean text
        if text.startswith("```json"):
            text = re.sub(r"^```json", "", text)
            text = re.sub(r"```$", "", text)
        elif text.startswith("```"):
            text = text[3:]
            text = text[:-3] if text.endswith("```") else text
        # parse json
        try:
            return json.loads(text)  
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text) 
            except Exception as ex:
                print(f"[Index {idx}] JSON Parse Error: {ex}")
                return None
    except Exception as e:
        print(f"[Index {idx}] JSON Parse Error: {e}")
        return None

def compute_error_label_weights(
    dist_df: pd.DataFrame,
    difficulty_dict: dict = None,
    importance_dict: dict = None,
    min_boost: float = 1.2,
    max_boost: float = 1.8,
    save_weights_path: str = '',
    prt: bool = False
):
    # higher weights to difficult or important labels
    difficulty_dict = difficulty_dict or {}
    importance_dict = importance_dict or {}

    adjusted_weights = []
    detail_rows = []

    for _, row in dist_df.iterrows():
        base_weight = row["proportion"]
        label = row["label"]

        difficulty_factor = difficulty_dict.get(label, 1.0)
        importance_factor = importance_dict.get(label, 1.0)

        freq_boost = 1.0
        if row["count"] < dist_df["count"].mean():
            freq_boost = random.uniform(min_boost, max_boost)

        final_weight = base_weight * difficulty_factor * importance_factor * freq_boost
        adjusted_weights.append(final_weight)

        detail_rows.append({
            "label": label,
            "base_proportion": base_weight,
            "count": row["count"],
            "difficulty": difficulty_factor,
            "importance": importance_factor,
            "freq_boost": round(freq_boost, 3),
            "final_weight_raw": final_weight 
        })

    # nomalization
    norm_weights = np.array(adjusted_weights) / sum(adjusted_weights)

    for i, row in enumerate(detail_rows):
        row["final_weight_norm"] = round(norm_weights[i], 5)
    weights_df = pd.DataFrame(detail_rows)
    if prt:
        print(weights_df.sort_values("final_weight_norm", ascending=False))

    if save_weights_path:
        weights_df.to_excel(save_weights_path, index=False)

    # sample one label
    sampled_label = np.random.choice(dist_df["label"], p=norm_weights)
    return sampled_label

def get_label_distribution(df, label_col: str = "error_label"):
    label_counter = Counter()
    for label_str in df[label_col]:
        if pd.isna(label_str):
            continue
        labels = [l.strip() for l in label_str.split(";")]
        label_counter.update(labels)
    total_labels = sum(label_counter.values())
    dist_df = pd.DataFrame(
        [(label, count, count / total_labels) for label, count in label_counter.items()],
        columns=["label", "count", "proportion"]
    )
    # sort by count
    dist_df = dist_df.sort_values(by="count", ascending=False).reset_index(drop=True)
    return dist_df

def load_prompt_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def format_few_shot(example_row):
    return f"""---
        Passage: {example_row['passage']}
        Answer: {example_row['answer']}
        Question: {example_row['question']}
        Error_label: {example_row['error_label']}
    ---"""

def create_prompt_with_few_shots(sample_df, passage, answer, random=True, difficulty_dict=None, importance_dict=None, assigned_label=None, label_col='error_label', template_path='./prompts/error_gen.txt'):
    template = load_prompt_template(template_path)
 
    if random:
        # random 
        example = sample_df.sample(1).iloc[0]
        few_shot_block = format_few_shot(example)
    else:
        # assign specific label (difficult labels, important labels) 
        dist_df = get_label_distribution(sample_df, label_col)
        if not assigned_label:
            assigned_label = compute_error_label_weights(dist_df, difficulty_dict, importance_dict, prt=False)
        assigned_df = sample_df[sample_df[label_col].str.contains(assigned_label)]
        example = assigned_df.sample(1).iloc[0]
        few_shot_block = format_few_shot(example)

    # replace slot of the prompt template
    error_types = {
        'Incomplete': '- Incomplete: Missing essential components, making the question unfinished.',
        'Not A Question': '- Not A Question: Lacks an interrogative structure or is a statement rather than a question.',
        'Spell Error': '- Spell Error: Contains misspelled words affecting readability or clarity.',
        'Grammar Error': '- Grammatical Error: Has grammatical issues such as incorrect word order, tense, or subject-verb agreement.',
        'Vague': '- Vague: The question is unclear, overly broad, or open to multiple interpretations due to vague language, making it difficult to determine the exact information being sought.',
        'Off Topic': '- Off Topic: The question is unrelated to the main topic of the passage',
        'Factual Error': '- Factual Error: Includes incorrect facts that contradict the passage.',
        'Information Not Mentioned': '- Information Not Mentioned: Asks for information not present in the passage.',
        'Unnecessary Copy from Passage': '- Unnecessary Copy from Passage: Copies too much text directly from the passage, making it redundant.',
        'Off Target Answer': '- Off Target Answer: Does not align with the provided answer.',
        'No Error': '- No Error: The question is clear, relevant, and answerable without any issues.',
    }
    error_label_list = [x.strip() for x in example[label_col].split(';')]
    error_info = '\n'.join([error_types[e] for e in error_label_list])
    final_prompt = template.replace('{{FEW_SHOT_EXAMPLES}}', few_shot_block).replace('{{ERROR}}', error_info).replace('{{PASSAGE}}', passage).replace('{{ANSWER}}', answer)
    return final_prompt, error_label_list


def batch_gen(model, df_candidate, sample_df, save_path, random=True, difficulty_dict=None, importance_dict=None, assigned_label=None, label_col='error_label', retry=3, prt=False):
    data = []
    for idx, row in tqdm(df_candidate.iterrows(), total=len(df_candidate)):
        prompt, error_label_list = create_prompt_with_few_shots(
            sample_df, row['passage'], row['answer'], random,
            difficulty_dict, importance_dict, assigned_label=assigned_label,
            label_col=label_col
        )
        messages = [
            {'role': 'system', 'content': 'You are a dataset generator for training an error identification model in question generation.'},
            {'role': 'user', 'content': prompt}
        ]
        if prt:
            print(prompt)
        example = ''
        for i in range(retry):
            res_text = completion_gpt(model, messages, prt=prt)
            example = safe_parse_json(res_text, idx)
            if not example:
                continue  # fail to parse json
            question = example.get('question')
            answer = row['answer']
            passage = row['passage'] 
            pred_label = example.get('error_label')
            # examine
            if not question or not pred_label:
                continue  # fail
            pred_error_list = [x.strip() for x in pred_label.split(';')]
            list1 = [x.lower() for x in error_label_list]
            list2 = [x.lower() for x in pred_error_list]
            if set(list1) != set(list2) or len(list1) != len(list2):
                continue  # fail
            # pass the examine, then add result
            data.append({
                'passage': passage,
                'answer': answer,
                'question': question,
                'source': row['source'].split('_')[0]+'_GPT-4',
                'error_label': pred_label
            })
            break
        if idx == 0:
            print(prompt)
            print(data)
        # save result 
        if idx % 5 == 0:
            pd.DataFrame(data).to_excel(save_path, index=False)
    df_all = pd.DataFrame(data)
    df_all.to_excel(save_path, index=False)
    print(f"Save {len(df_all)} samples to {save_path}")


    def review_result(model, cand_df, review_col, save_path, template_path='./prompts/error_gen_check.txt', prt=False):
        new_data = []
        for idx, row in tqdm(cand_df.iterrows(), total=len(cand_df)):
            prompt = load_prompt_template(template_path).replace('{{PASSAGE}}', row['passage']).replace('{{ANSWER}}', row['answer']).replace('{{QUESTION}}',row['question']).replace('{{ERROR}}',row[review_col])
            messages = [
                {'role': 'system', 'content': 'You are an error label verifier.'},
                {'role': 'user', 'content': prompt}
            ]
           
            if 'gemini' in model:
                res = apply_gemini(model, prompt, prt=prt)
            elif 'gpt' in model:
                res = completion_gpt(model, messages, prt=prt)
            elif 'claude' in model:
                res = apply_claude(model, messages, prt=prt)
            else:
                print('Model {} not support!'.format(model))     
                break

            if prt:
                print('='*50)
                print(prompt)
                print(res)
                print('='*50)
            one = row.to_dict()
            one['check-'+model.split('-')[0]] = float(res)
            new_data.append(one)
            if idx % 2 == 0:
                pd.DataFrame(new_data).to_excel(save_path, index=False)
        df_all = pd.DataFrame(new_data)
        df_all.to_excel(save_path, index=False)
        print(f"Save {len(df_all)} Samples to {save_path}")


    def filter_by_score(df, model_cols, strategy='vote', threshold=0.8, low_precision_labels=None, by='vote', label_col='error_label', save_path=None, return_with_score=False):
        df = df.copy()

        def get_vote_requirement(row, by='vote'):
            if by != 'vote':
                return 2
            labels = [lbl.strip() for lbl in row[label_col].split(';')]
            return 3 if any(lbl in low_precision_labels for lbl in labels) else 2
        
        def get_thes_requirement(row, threshold=0.8, by='threshold'):
            if by != 'threshold':
                return threshold
            labels = [lbl.strip() for lbl in row[label_col].split(';')]
            return 0.9 if any(lbl in low_precision_labels for lbl in labels) else threshold
        
        df['required_thres'] = df.apply(lambda row: get_thes_requirement(row, threshold, by), axis=1)
        df['required_vote'] = df.apply(lambda row: get_vote_requirement(row, by), axis=1)

        if strategy == 'vote':
            df['vote'] = (df[model_cols].values >= df['required_thres'].values[:, None]).sum(axis=1)
            result_df = df[df['vote'] >= df['required_vote']]
            # label distribution
            all_labels = []
            for lbl_str in result_df[label_col]:
                all_labels.extend([lbl.strip() for lbl in lbl_str.split(';') if lbl.strip()])
            label_counts = Counter(all_labels)
            print("label distribution:")
            for label, count in label_counts.items():
                print(f"{label:<20} : {count}")
            if not return_with_score:
                result_df = result_df.drop(columns=['vote', 'required_vote', 'required_thres'])
            
        elif strategy == 'avg':
            df['avg_score'] = df[model_cols].mean(axis=1)
            result_df = df[df['avg_score'] >= threshold]

            all_labels = []
            for lbl_str in result_df[label_col]:
                all_labels.extend([lbl.strip() for lbl in lbl_str.split(';') if lbl.strip()])
            label_counts = Counter(all_labels)

            print("label distribution:")
            for label, count in label_counts.items():
                print(f"{label:<20} : {count}")

            if not return_with_score:
                result_df = result_df.drop(columns=['avg_score'])

        else:
            raise ValueError("strategy can noly be 'vote' or 'avg'")

        if save_path:
            result_df.to_excel(save_path, index=False)
            print("Origin data count= {}, filtered data count = {}, save result to {}".format(
                len(df), len(result_df), save_path
            ))
        return result_df




if __name__ == "__main__":
    # generate error questions given specific error types
    model = 'gpt-4o'
    sample_df = pd.read_excel('../../data/samples_for_simulation.xlsx') 
    # sample_df = sample_df[sample_df['error_label']!='No Error']
    print('sample count = {}'.format(len(sample_df)))
    # unlabeled pool
    df_candidate = pd.read_excel('../../data/filter.xlsx')
    df_candidate = df_candidate.iloc[0:20000]
    save_path = '../../data/iter0/simulation.xlsx'
    random = False
    # difficult to identify
    difficulty_dict = {
        "Factual Error": 1.4,                  
        "Information Not Mentioned": 1.4,      
        "Grammar Error": 1.2,                  
        "Off Target Answer": 1.4,
        "Vague": 1.6,
        "Not A Question": 1.2,                 
    }
    # importance weights
    importance_dict = {
        "Factual Error": 1.4,                    # Factual error, critical and high priority
        "Off Target Answer": 1.4,                # Answer deviates from the intended target, highly impactful
        "Information Not Mentioned": 1.4,        # Missing information, affects understanding
        "Vague": 1.4,                            # Vague expression, potentially misleading
        "Not A Question": 1.2,                   # Improper question format
        "Incomplete": 1.0,                       # Incomplete question, relatively common issue
        "Spell Error": 0.8,                      # Minor impact
        "Off Topic": 0.6,                        # Rarely occurs
    }
    batch_gen(model, df_candidate, sample_df, save_path, 
            difficulty_dict, importance_dict, assigned_label=None,
            label_col='error_label',retry=3, prt=False)


    # # check generated data by llms
    # model = 'gpt-4o'
    # # model = "claude-3-5-haiku-20241022"
    # template_path='./prompts/label/error_gen_check.txt'
    # data_path = '../../data/iter0/simulation.xlsx'
    # cand_df = pd.read_excel(data_path)
    # save_path = data_path.replace('.xlsx', '-check.xlsx')
    # review_result(model, cand_df, 'error_label', save_path, template_path=template_path, prt=True)


    # # filter by score
    # data_path = '../../data/iter0/simulation-check-all.xlsx'
    # save_path = data_path.replace('.xlsx','-filter.xlsx')
    # df = pd.read_excel(data_path)
    # model_cols = ['check-gemini', 'check-gpt', 'check-claude']
    # # low_precision_labels = {}
    # low_precision_labels = {'Vague', 'Factual Error', 'Information Not Mentioned', 'Off Target Answer'}
    # filtered_df = filter_by_score(df, model_cols, strategy='vote', threshold=0.8, low_precision_labels=low_precision_labels, by='threshold', label_col='error_label', save_path=save_path, return_with_score=False)
    # print(filtered_df.shape)