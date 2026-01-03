# Repository for ErrEval

## Abstract
Automatic Question Generation (QG) often produces outputs containing critical defects, such as factual hallucinations and answer mismatches. However, existing evaluation methods, including LLM-based evaluators, tend to overlook such defects, leading to overestimated question quality due to their reliance on black-box and holistic scoring without explicit error modeling.
To address this limitation, we propose ErrEval, a flexible and Error-aware Evaluation framework that enhances QG evaluation through explicit error diagnostics. ErrEval reformulates evaluation as a two-stage process, consisting of error diagnosis followed by informed scoring.
At the core of ErrEval is a lightweight, plug-and-play Error Identifier that detects and categorizes common error types across structural, linguistic, and content-related aspects. The resulting diagnostic signals serve as explicit evidence to guide LLM evaluators toward more fine-grained and grounded judgements.
Experiments show that a RoBERTa-based Error Identifier trained with an iterative strategy outperforms zero-shot LLM-based baselines by up to 27.7\% in relative micro-F1. Moreover, integrating ErrEval into multiple LLM evaluators improves their alignment with human judgments and reduces the overestimation of low-quality questions.

## Framework
[framework](./figures/framework.pdf)

## Usage
For usage instructions, please refer to the following files.

- [Code](./code)
- [Data](./data)
- [Model](./model)
- [Result](./result)

## Citation
