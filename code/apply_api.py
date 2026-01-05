import time
import numpy as np
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
import re
from utils import read_text, get_score, load_json
import anthropic
import os



def completion_gpt(model, messages, base_url, api_key, max_try=3, prt=False):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
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


def apply_claude(model, messages, api_key, max_tokens=256, max_try=3, prt=False):
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=api_key,
    )
    # try again when fail
    for i in range(max_try):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
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


class ApplyAPI:
    def __init__(self, config_path):
        self.template = None
        self.config = load_json(config_path)
    
    def get_error_info(self, dimension, error): 
        if not error:
            return ''
        error_types = {
            'Incomplete': '- Incomplete: Missing essential components, making the question unfinished.',
            'Not A Question': '- Not A Question: Lacks an interrogative structure or is a statement rather than a question.',
            'Spell Error': '- Spell Error: Contains misspelled words affecting readability or clarity.',
            'Grammatical Error': '- Grammatical Error: Has grammatical issues such as incorrect word order, tense, or subject-verb agreement.',
            'Vague': '- Vague: The question is unclear, overly broad, or open to multiple interpretations due to vague language, making it difficult to determine the exact information being sought.',
            'Off Topic': '- Off Topic: The question is unrelated to the main topic of the passage',
            'Factual Error': '- Factual Error: Includes incorrect facts that contradict the passage.',
            'Information Not Mentioned': '- Information Not Mentioned: Asks for information not present in the passage.',
            'Unnecessary Copy from Passage': '- Unnecessary Copy from Passage: Copies too much text directly from the passage, making it redundant.',
            'Off Target Answer': '- Off Target Answer: Does not align with the provided answer.',
            'No Error': '- No Error: The question is clear, relevant, and answerable without any issues.',
        }
        dimension_to_error = {
            'fluency': ['Incomplete', 'Spell Error', 'Grammatical Error', 'No Error'],
            'clarity': ['Incomplete', 'Not A Question', 'Grammatical Error', 'Vague', 'No Error'],
            'conciseness': ['Unnecessary Copy from Passage', 'No Error'],
            'relevance': ['Off Topic', 'No Error'],
            'consistency': ['Off Topic', 'Factual Error', 'Information Not Mentioned', 'No Error'],
            'answerability': ['Incomplete', 'Not A Question', 'Vague', 'Off Topic', 'Factual Error', 'Information Not Mentioned', 'No Error'],
            'answer_consistency': ['Incomplete', 'Not A Question', 'Vague', 'Off Topic', 'Factual Error', 'Information Not Mentioned', 'Off Target Answer', 'No Error']
        }
        rel_errors = dimension_to_error[dimension]
        identified_errors = [x.strip() for x in error.split(';')]
        matched_errors = [e for e in identified_errors if e in rel_errors]
        error_info = '\n'.join([error_types[e] for e in matched_errors])
        return error_info

    def get_messages(self, p, q, a, dimension, error=None, prt=False):
        role = 'You are an expert in evaluating the quality of generated questions.'
        prompt = self.template.replace('#passage',p).replace('#answer',a).replace('#question',q)
        if error:
            error_info = self.get_error_info(dimension, error)
            prompt = self.template.replace('#passage',p).replace('#answer',a).replace('#question',q).replace('#error',error_info)
        if prt:
            print(prompt)
        messages = [
            {'role': 'system', 'content': role},
            {'role': 'user', 'content': prompt}
        ]
        return messages

    def request_one(self, model, p, q, a, dimension, template_path, error=None, prt=False):
        res_message, score = None, 999
        if not self.template:
            self.template = read_text(template_path)
        messages = self.get_messages(p, q, a, dimension, error, prt)
        if 'claude' in model:
            res_message = apply_claude(
                model, messages, 
                api_key=self.config.get('api_key'), 
                prt=prt
            )
        else:
            res_message = completion_gpt(
                model, messages, 
                base_url=self.config.get('base_url'), 
                api_key=self.config.get('api_key'), 
                prt=prt
            )
        
        score = get_score(res_message)
        return res_message, score

    def request_batch(self, model, dimension, data, template_path, error_col=None, save_path=None, need_rational=False, prt=False):
        new_data = []
        new_col = dimension+'_'+model
        score_col = dimension+'_'+model+'_score'
        error = None
        for idx, item in tqdm(enumerate(data), total=len(data)): 
            if error_col:
                error = item[error_col]
            message, score = self.request_one(model, item['passage'], item['question'], item['answer'], dimension, template_path, error=error, need_rational=need_rational, prt=prt)
            item[new_col] = message
            item[score_col] = score
            new_data.append(item)
            if idx == 0:
                print('#'*10, 'Prompt', '#'*10)
                print(self.get_messages(item['passage'],item['question'],item['answer'],dimension,error,prt=prt))
                print('#'*10, 'Result', '#'*10)
                print('Message: ', message)
                print('Score: ', score)
            if idx % 5 == 0 and save_path is not None:
                pd.DataFrame(new_data).to_excel(save_path, index=False)
        if save_path is not None:
            pd.DataFrame(new_data).to_excel(save_path, index=False)
        return new_data
    

if __name__ == "__main__":
    model = 'gpt-4o'
    # model = 'claude-3-5-haiku-20241022'
    dimension = 'answer_consistency'
    error_col = 'pred_error'
    config_path = './config.json'
    cgpt = ApplyAPI(config_path)
    template_path = os.path.join(cgpt.config.get("prompt_dir"), dimension+'.txt')
    if error_col:
        template_path = os.path.join(cgpt.config.get("prompt_dir"), dimension+'_error.txt')
    

    # batch request
    iter_count = 3
    model_type = 'roberta'
    model_size = 'base'
    data_path = '../result/test/iter{}/{}-{}/qgeval-el.xlsx'.format(str(iter_count), model_type, model_size)
    save_path = '../result/test/{}/iter{}/{}-{}/qgeval-{}-el.xlsx'.format(model.split('-')[0], str(iter_count), model_type, model_size, dimension)
    df = pd.read_excel(data_path)
    df = df.fillna('')
    data = df.to_dict(orient='records')
    print(len(data))
    new_data = cgpt.request_batch(model, dimension, data, template_path, error_col=error_col, save_path=save_path, need_rational=True, prt=False)
    