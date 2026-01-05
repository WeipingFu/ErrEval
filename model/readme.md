This directory stores the downloaded model files.

## Model Files
Download the models you need:
- Error Identifier: (1) [base model](https://huggingface.co/anonymous-11/erreval-base/tree/main); (2) [large model](https://huggingface.co/anonymous-11/erreval-large).
- Evaluator: download corresponding models from [Hugging Face](https://huggingface.co/models) 


## Usage
- Error Identifier: dectect error types in generated questions. For use, update fields in `../code/config.json`.

```json
{
    "prompt_dir": "./prompts/error_label/",
    "error_aware": true,
    "error_model_path": "../model/error_label/erreval_base",
    "error_tokenizer_path": "../model/error_label/erreval_base"
}
```
