# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from utils import load_json, get_score, ensure_directory_exists
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd


def qwen_one(model, tokenizer, messages, max_new_tokens=256, enable_thinking=False, device='cuda'):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    score = get_score(content)
    return content, score, thinking_content


def predict(model_path, test_data_path, max_new_tokens=256, col_name='qwen', enable_thinking=False, res_save_path=None):
    new_data = []
    test_data = load_json(test_data_path)
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    device = model.device
    print('device:', device)
    for example in tqdm(test_data):
        content, score, thinking_content = qwen_one(model, tokenizer, example['messages'], max_new_tokens, enable_thinking, device)
        example[col_name] = content
        example[col_name+'_score'] = score
        if enable_thinking:
            example[col_name+'_thinking'] = thinking_content
        new_data.append(example)
        if len(new_data) == 1:
            for k, v in example.items():
                print(k, v)
        # if len(new_data) % 2 == 0 and res_save_path is not None:
        #     pd.DataFrame(new_data).to_excel(res_save_path, index=False)
    if res_save_path is not None:
        output_dir = os.path.dirname(res_save_path)
        ensure_directory_exists(output_dir)
        pd.DataFrame(new_data).to_excel(res_save_path, index=False)


if __name__ == "__main__":
    model_path = '../model/pretrained_model/qwen3-8b'
    dimension = 'fluency'
    model_type = 'roberta'
    model_size = 'base'
    iter_count = 3
    test_data_path = '../data/iter{}/{}-{}/qgeval-{}-el.json'.format(str(iter_count), model_type, model_size, dimension)
    res_save_path = '../result/qwen3/iter{}/{}-{}/qgeval-{}.xlsx'.format(str(iter_count), model_type, model_size, dimension)
    # predict(model_path, test_data_path, max_new_tokens=5120, col_name='answer_consistency_qwen',enable_thinking=True, res_save_path=res_save_path)
    predict(model_path, test_data_path, max_new_tokens=256, col_name='{}_qwen'.format(dimension),enable_thinking=False, res_save_path=res_save_path)


