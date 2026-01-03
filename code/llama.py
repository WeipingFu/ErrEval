# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'

from utils import load_json, get_score, ensure_directory_exists
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)
import torch
from tqdm import tqdm
import pandas as pd


def llama_one(model, tokenizer, messages, max_new_tokens=256, device='cuda'):
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
    )
    output_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    score = get_score(output_text)
    return output_text, score

# batch predict
def predict(model, tokenizer, test_data_path, max_new_tokens=256, col_name='llama3', res_save_path=None, device='cuda'):
    new_data = []
    test_data = load_json(test_data_path)
    for example in tqdm(test_data):
        output_text, score = llama_one(model, tokenizer, example['messages'], max_new_tokens=max_new_tokens, device=device)
        example[col_name] = output_text
        example[col_name+'_score'] = score
        new_data.append(example)
        if len(new_data) == 1:
            for k, v in example.items():
                print(k, v)
    if res_save_path is not None:
        output_dir = os.path.dirname(res_save_path)
        ensure_directory_exists(output_dir)
        pd.DataFrame(new_data).to_excel(res_save_path, index=False)



if __name__ == "__main__":
    
    # predict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    model = AutoModelForCausalLM.from_pretrained(
        '../model/pretrained_model/llama-3-8b-instruct',
        torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained('../model/pretrained_model/llama-3-8b-instruct')
    tokenizer.add_special_tokens({"pad_token": "<PAD>",})
    LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    model_type = 'roberta'
    model_size = 'base'
    dimension = 'fluency'
    iter_count = 3
    test_data_path = '../data/iter{}/{}-{}/qgeval-{}-el.json'.format(str(iter_count), model_type,model_size,dimension)
    res_save_path = '../result/llama3/iter{}/{}-{}/qgeval-{}.xlsx'.format(str(iter_count), model_type,model_size,dimension)
    predict(model, tokenizer, test_data_path, max_new_tokens=256, col_name='{}_llama3'.format(dimension), res_save_path=res_save_path, device=device)

