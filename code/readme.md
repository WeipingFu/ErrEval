This is code for ErrEval

# Installation
Download the code, create an enviroment run in `Python 3.11.5`, and install the packages in `requirements.txt`.

Download the models you need:
- Error Identifier: (1) [base model](https://huggingface.co/anonymous-11/erreval-base/tree/main); (2) [large model](https://huggingface.co/anonymous-11/erreval-large).
- Evaluator: download corresponding models from [Hugging Face](https://huggingface.co/models) 

# Run for evaluation
Run `eval.py` to start the evaluation.
Modify `config.json` to configure the evaluator and error identifier settings accordingly. 

```python
    p = 'Graptopetalum (leatherpetal) is a plant genus of the family "Crassulaceae".  They are perennial succulent plants and native to Mexico and Arizona.  They grow usually in a rosette.  There are around 19 species in this genus.&#10;Couroupita is a genus of flowering plants of Lecythidaceae family first described as a genus in 1775.  It is native to tropical South America and Central America.'
    q = 'Are Graptopetalum plants native to Mexico and Arizona?'
    a = 'yes'
    dimension = 'answerability'
    config_path = './config.json'
    erreval = ErrEval(config_path)
    print(erreval.eval(p, q, a, dimension))
```
##  Configuration Guide for `config.json`
This evaluation framework supports multiple evaluators and error identifier models. You can flexibly configure them via the `config.json` file. Below is a sample configuration along with detailed explanations of each field.
```json
{
    "evaluator_name": "gpt-4o",   
    "evaluator_type": "close",
    "evaluator_path": "../model/pretrained_model/llama-3-8b-instruct",
    "tokenizer_path": "../model/pretrained_model/llama-3-8b-instruct",
    "prompt_dir": "./prompts/direct/",
    "error_aware": false,
    "error_model_path": "../model/error_label/roberta-base",
    "error_tokenizer_path": "../model/pretrained_model/roberta-base",
    "max_new_tokens": 256,
    "enable_thinking": false,
    "do_sample": true
}
```
### Field Descriptions

| Field Name            | Type     | Description                                                                 |
|------------------------|----------|-----------------------------------------------------------------------------|
| `evaluator_name`      | `string` | The name of the evaluator (e.g., `"gpt-4o"`, `llama`).                               |
| `evaluator_type`      | `string` | Evaluation type, typically `"close"` or `"open"`.                          |
| `evaluator_path`      | `string` | Path to the local or remote evaluator model. Ignored if `evaluator_type` is `"close"`.                              |
| `tokenizer_path`      | `string` | Path to the tokenizer corresponding to the evaluator. Ignored if `evaluator_type` is `"close"`.                     |
| `prompt_dir`          | `string` | Directory containing prompt templates.                                     |
| `error_aware`         | `bool`   | Whether to enable error-aware evaluation.                                  |
| `error_model_path`    | `string` | Path to the error identifier model (e.g., a classifier). Ignored if `error_aware` is `"false"`.                   |
| `error_tokenizer_path`| `string` | Path to the tokenizer for the error identifier model. Ignored if `error_aware` is `"false"`.                      |
| `max_new_tokens`      | `int`    | The maximum number of tokens to generate during evaluation.                |
| `enable_thinking`     | `bool`   | Whether to enable "thinking" mode.      |
| `do_sample`           | `bool`   | Whether to use sampling during generation (if `false`, greedy decoding).   |


Note: for close-source evalutors, please modify the `apply_api.py` to set your api_key or add new models except for `gpt` and `claude` to conduct evaluation. 


## Other Code Files
- `multi_classify.py`: Used to train or evaluate the error identifier.
- `verify.py`: Used to train or evaluate the verifier.
- `prompt_data.py`: Generates test files for evaluation.
- `llama.py`: Uses LLaMA as an evaluator.
- `qwen.py`: Uses Qwen as an evaluator.
- `apply_api.py`: Uses close-source models as evaluators.
- `utils.py`: Contains common utility functions.
- `tools/simulate_error.py`: Generates questions that contain errors.
- `tools/corr.py`: Computes the Pearson correlation coefficient.
