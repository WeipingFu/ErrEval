import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM
)
from sklearn.preprocessing import MultiLabelBinarizer
from utils import load_json, get_score
from prompt_data import append_messages_dimension_one, append_messages_with_error_label_one
from llama import llama_one
from qwen import qwen_one
from apply_api import ApplyAPI


class ErrEval:
    def __init__(self, config_path):
        # load config file
        self.config = load_json(config_path)

    # load Error Identifier
    def load_ei_model(self, device):
        LABEL_LIST = ['Incomplete', 'Not A Question', 
              'Spell Error', 'Grammar Error', 'Vague', 'Unnecessary Copy from Passage', 
              'Off Topic', 'Factual Error', 'Information Not Mentioned', 'Off Target Answer',  
              'No Error']
        self.mlb = MultiLabelBinarizer(classes=LABEL_LIST)
        self.mlb.fit([[label] for label in LABEL_LIST])   
        self.ei_tokenizer = AutoTokenizer.from_pretrained(self.config['error_tokenizer_path'])
        self.ei_model = AutoModelForSequenceClassification.from_pretrained(self.config['error_model_path']).to(device)
    
    # load evaluator
    def load_evaluator(self, device):
        self.evaluator, self.tokenizer = None, None
        evaluator_type = self.config['evaluator_type']
        if evaluator_type == 'open':
            self.evaluator = AutoModelForCausalLM.from_pretrained(self.config['evaluator_path']).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer_path'])
        elif evaluator_type == 'close':
            self.evaluator = ApplyAPI()
        else:
            print('Model type {} is not supported!'.format(evaluator_type))

        if 'llama' in self.config['evaluator_name']: 
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>",})
            LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
            self.tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

    # conduct error identification
    def error_identify(self, p, q, a, device):
        text = q + '</s>' + a + '</s>' + p
        inputs = self.ei_tokenizer(text, return_tensors="pt", max_length=512, padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # predict
        with torch.no_grad():
            outputs = self.ei_model(**inputs)
            logits = outputs.logits.cpu().numpy()
        
        probs = 1 / (1 + np.exp(-logits)) 
        num_labels = probs.shape[1]
        thresholds = np.array([0.5] * num_labels)
        pred_binary = (probs > thresholds).astype(int)
        pred_labels = self.mlb.inverse_transform(pred_binary)
        return '; '.join(pred_labels[0]) if pred_labels[0] else ''
    
    # open evaluator predict
    def generate_and_score(self, messages, device, **kwargs):
        chat_template_args = {
            "messages": messages,
            "add_generation_prompt": True,
            "return_tensors": "pt"
        }

        model_inputs = self.tokenizer.apply_chat_template(**chat_template_args)

        if device is None:
            device = self.evaluator.device
        model_inputs = model_inputs.to(device)

        eos_token_ids = kwargs.pop("eos_token_id", [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ])
        pad_token_id = kwargs.pop("pad_token_id", self.tokenizer.pad_token_id)

        # generate
        outputs = self.evaluator.generate(
            **model_inputs,
            max_new_tokens=self.config['max_new_tokens'],
            eos_token_id=eos_token_ids,
            pad_token_id=pad_token_id,
            do_sample=self.config['do_sample'],
            **kwargs
        )
        output_text = self.tokenizer.decode(
            outputs[0][model_inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()
        score = get_score(output_text)
        return output_text, score
    
    # apply evaluation
    def eval(self, p, q, a, dimension):
        reponse, score, thinking = None, 999, None
        error_label = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # apply prompt template
        if self.config['error_aware']:
            self.load_ei_model(device)
            error_label = self.error_identify(p, q, a, device)
            messages = append_messages_with_error_label_one(p, q, a, error_label, dimension, prompt_dir=self.config['prompt_dir'])
        else:
            messages = append_messages_dimension_one(p, q, a, dimension, prompt_dir=self.config['prompt_dir'])
        
        # load evaluator
        self.load_evaluator(device)

        print('Messages:')
        for one in messages:
            print(one)
        print('\n')
        
        # evaluate
        evaluator_type = self.config['evaluator_type']
        evaluator_name = self.config['evaluator_name']

        if evaluator_type == 'open':
            if 'llama' in evaluator_name:
                reponse, score = llama_one(self.evaluator, self.tokenizer, messages, max_new_tokens=self.config['max_new_tokens'], device=device)
            elif 'qwen' in evaluator_name:
                reponse, score, thinking = qwen_one(self.evaluator, self.tokenizer, messages, max_new_tokens=self.config['max_new_tokens'], enable_thinking=self.config['enable_thinking'], device=device)
            else:
                reponse, score = self.generate_and_score(messages, device)
        elif evaluator_type == 'close':
            template_path = self.config['prompt_dir']+'/{}.txt'.format(dimension)
            reponse, score = self.evaluator.request_one(evaluator_name, p, q, a, dimension, template_path, error=error_label, prt=False)
        else:
            print('Model type {} is not supported!'.format(evaluator_type))
        
        return reponse, score
    
    

if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    p = 'Graptopetalum (leatherpetal) is a plant genus of the family "Crassulaceae".  They are perennial succulent plants and native to Mexico and Arizona.  They grow usually in a rosette.  There are around 19 species in this genus.&#10;Couroupita is a genus of flowering plants of Lecythidaceae family first described as a genus in 1775.  It is native to tropical South America and Central America.'
    q = 'Are Graptopetalum plants native to Mexico and Arizona?'
    a = 'yes'
    dimension = 'answerability'
    config_path = './config.json'
    erreval = ErrEval(config_path)
    print(erreval.eval(p, q, a, dimension))