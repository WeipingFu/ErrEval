from utils import read_text, save_json, load_json
import pandas as pd
import os

def append_messages_with_error_label_one(p, q, a, error_label, dimension, prompt_dir='./prompts/error_label/'):
    error_types = {
        'Incomplete': '- Incomplete: Missing essential components, making the question unfinished.',
        'Not A Question': '- Not A Question: Lacks an interrogative structure or is a statement rather than a question.',
        'Spell Error': '- Spell Error: Contains misspelled words affecting readability or clarity.',
        'Grammar Error': '- Grammar Error: Has grammatical issues such as incorrect word order, tense, or subject-verb agreement.',
        'Vague': '- Vague: The question is unclear, overly broad, or open to multiple interpretations due to vague language, making it difficult to determine the exact information being sought.',
        'Off Topic': '- Off Topic: The question is unrelated to the main topic of the passage',
        'Factual Error': '- Factual Error: Includes incorrect facts that contradict the passage.',
        'Information Not Mentioned': '- Information Not Mentioned: Asks for information not present in the passage.',
        'Unnecessary Copy from Passage': '- Unnecessary Copy from Passage: Copies too much text directly from the passage, making it redundant.',
        'Off Target Answer': '- Off Target Answer: Does not align with the provided answer.',
        'No Error': '- No Error: The question is clear, relevant, and answerable without any issues.',
    }
    dimension_to_error = {
        'fluency': ['Incomplete', 'Spell Error', 'Grammar Error', 'No Error'],
        'clarity': ['Incomplete', 'Not A Question', 'Grammar Error', 'Vague', 'No Error'],
        'conciseness': ['Unnecessary Copy from Passage', 'No Error'],
        'relevance': ['Off Topic', 'No Error'],
        'consistency': ['Off Topic', 'Factual Error', 'Information Not Mentioned', 'No Error'],
        'answerability': ['Incomplete', 'Not A Question', 'Vague', 'Off Topic', 'Factual Error', 'Information Not Mentioned', 'No Error'],
        'answer_consistency': ['Incomplete', 'Not A Question', 'Vague', 'Off Topic', 'Factual Error', 'Information Not Mentioned', 'Off Target Answer', 'No Error']
    }
    prompt_path = os.path.join(prompt_dir, dimension+'_error.txt')
    prompt_template = read_text(prompt_path)
    rel_errors = dimension_to_error[dimension]
    identified_errors = [x.strip() for x in error_label.split(';')]
    matched_errors = [e for e in identified_errors if e in rel_errors]
    error_information = '\n'.join([error_types[e] for e in matched_errors])
    prompt = prompt_template.replace('#passage',p).replace('#answer',a).replace('#question',q).replace('#error', error_information)
    messages = [
        {'role': 'system', 'content': 'You are an expert in evaluating the quality of generated questions.'},
        {'role': 'user', 'content': prompt}
    ]
    return messages


def append_messages_with_error_label(data_path, res_col, dimension, save_path='', prompt_dir='./prompts/error_label/', prt=False):
    # example of element in res_col: Incomplete; Not A Question
    if '.xlsx' in data_path:
        df = pd.read_excel(data_path)
        df = df.fillna('')
        data = df.to_dict(orient='records')
    elif '.json' in data_path:
        data = load_json(data_path)
    else:
        print('Data format is not supported, fail to load data!')
        return 
    
    new_data = []
    for one in data:
        messages = append_messages_with_error_label_one(one['passage'], one['question'], one['answer'], one[res_col], dimension, prompt_dir)
        one['messages'] = messages
        new_data.append(one)
    print('total data count = {}'.format(len(new_data)))
    if prt:
        for k, v in new_data[0].items():
            print(k, v)
    if save_path:
        if '.xlsx' in save_path:
            pd.DataFrame(new_data).to_excel(save_path, index=False)
        elif '.json' in save_path:
            save_json(new_data, save_path)
        else:
            print('Save format is not supported, fail to save data!')
    return new_data


def append_messages_dimension_one(p, q, a, dimension, prompt_dir='./prompts/direct/'):
    prompt_path = os.path.join(prompt_dir, dimension+'.txt')
    prompt_template = read_text(prompt_path)
    prompt = prompt_template.replace('#passage',p).replace('#answer',a).replace('#question',q)
    messages = [
        {'role': 'system', 'content': 'You are an expert in evaluating the quality of generated questions.'},
        {'role': 'user', 'content': prompt}
    ]
    return messages

def append_meessages_dimension(data_path, dimension, save_path='', prompt_dir='./prompts/direct/', prt=False):
    if '.xlsx' in data_path:
        data = pd.read_excel(data_path).to_dict(orient='records')
    elif '.json' in data_path:
        data = load_json(data_path)
    else:
        print('Data format is not supported, fail to load data!')
        return 
    new_data = []
    
    for one in data:
        messages = append_messages_dimension_one(one['passage'], one['question'], one['answer'], dimension, prompt_dir)
        one['messages'] = messages
        new_data.append(one)
    print('total data count = {}'.format(len(new_data)))
    if prt:
        for k, v in new_data[0].items():
            print(k, v)
    if save_path:
        if '.xlsx' in save_path:
            pd.DataFrame(new_data).to_excel(save_path, index=False)
        elif '.json' in save_path:
            save_json(new_data, save_path)
        else:
            print('Save format is not supported, fail to save data!')
    return new_data



if __name__ == '__main__':
    # # vanilla prompt
    # dimension = 'fluency'
    # data_path = '../data/test/qgeval.xlsx'
    # save_path = '../data/test/direct/qgeval-{}.xlsx'.format(dimension)
    # new_data = append_meessages_dimension(data_path, dimension, save_path=save_path, prt=True)


    # error-aware prompt
    iter_count = 3
    model_type = 'roberta'
    model_size = 'base'
    res_col = 'pred_error'
    dimension = 'answerability'
    data_path = '../result/iter{}/{}-{}/qgeval-el.xlsx'.format(str(iter_count), model_type, model_size)
    save_path = '../data/iter{}/{}-{}/qgeval-{}-el.json'.format(str(iter_count), model_type, model_size, dimension)
    new_data = append_messages_with_error_label(data_path, res_col, dimension, save_path=save_path, prompt_dir='./prompts/error_label/', prt=True)
