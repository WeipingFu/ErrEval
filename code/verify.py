# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, EarlyStoppingCallback)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import ensure_directory_exists

# settings
EPOCH = 30
MAX_LENGTH = 512

def get_sep_token(model_type):
    return {
        'bert': '[SEP]',
        'deberta': '[SEP]',
        'roberta': '</s>',
    }.get(model_type.lower(), '[SEP]')

# load data
def load_and_preprocess_data(path, res_col, model_type='roberta'):
    df = pd.read_excel(path)
    has_label = 'label' in df.columns
    if has_label:
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
    sep_token = get_sep_token(model_type)
    df['text'] = df[res_col].astype(str) + sep_token + \
                 df['question'].astype(str) + sep_token + \
                 df['answer'].astype(str) + sep_token + \
                 df['passage'].astype(str)
    
    if has_label:
        return df, df['text'].tolist(), df['label'].tolist()
    else:
        return df, df['text'].tolist()


# dataset
class BinaryDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH)
        self.labels = torch.tensor(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# metrics
def compute_metrics(pred):
    logits = pred.predictions
    preds = np.argmax(logits, axis=1)
    labels = pred.label_ids
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds),
    }

# predict and compute metrics
def predict_and_save(trainer, df, tokenizer, texts, labels, output_file="predictions.xlsx"):
    pred_output = trainer.predict(BinaryDataset(texts, labels, tokenizer))
    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()  # softmax
    preds = np.argmax(probs, axis=1)
    pos_probs = probs[:, 1]  

    df['verify_label'] = preds
    df['positive_prob'] = pos_probs  

    output_dir = os.path.dirname(output_file)
    ensure_directory_exists(output_dir)
    df.to_excel(output_file, index=False)
    print(f"Save results to {output_file}")

    print("Classification Report:")
    print("  Accuracy: ", round(accuracy_score(labels, preds), 3))
    print("  F1 Score: ", round(f1_score(labels, preds), 3))
    print("  Precision: ", round(precision_score(labels, preds), 3))
    print("  Recall: ", round(recall_score(labels, preds), 3))
    print(classification_report(labels, preds))


# train model
def train_model(train_path, model_name, res_col, model_type, dev_path=None, tokenizer_path=None, batch_size=32, output_dir="outputs", output_file="train_predictions.xlsx", dev_output_file="dev_predictions.xlsx"):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path else model_name)
    df, texts, labels = load_and_preprocess_data(train_path, res_col, model_type)

    # split train & dev
    if not dev_path:
        train_texts, dev_texts, train_labels, dev_labels, train_df, dev_df = train_test_split(
            texts, labels, df, test_size=0.2, random_state=42
        )
    else:
        dev_df, dev_texts, dev_labels = load_and_preprocess_data(dev_path, res_col, model_type)
        train_texts = texts
        train_df = df
        train_labels = labels
    print(f"Train size: {len(train_df)}, Dev size: {len(dev_df)}")

    train_dataset = BinaryDataset(train_texts, train_labels, tokenizer)
    dev_dataset = BinaryDataset(dev_texts, dev_labels, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    ensure_directory_exists(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=EPOCH,
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
    )

    early_stop = EarlyStoppingCallback(early_stopping_patience=3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stop]
    )

    trainer.model.to(device)
    trainer.train()
    trainer.evaluate()

    print('******** Start Train Predicting ********')
    predict_and_save(trainer, train_df, tokenizer, train_texts, train_labels, output_file)
    print('******** Start Dev Predicting ********')
    predict_and_save(trainer, dev_df, tokenizer, dev_texts, dev_labels, dev_output_file)


def predict_and_save_texts_only(test_path, model_path, tokenizer_path, res_col, output_file="predictions.xlsx", model_type='roberta'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    trainer = Trainer(model=model)

    df, texts = load_and_preprocess_data(test_path, res_col, model_type=model_type)
    dummy_labels = [0] * len(texts)
    dataset = BinaryDataset(texts, dummy_labels, tokenizer)

    pred_output = trainer.predict(dataset)
    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    pos_probs = probs[:, 1]
    # save results
    df["verify_label"] = preds
    df["positive_prob"] = pos_probs
    output_dir = os.path.dirname(output_file)
    ensure_directory_exists(output_dir)
    df.to_excel(output_file, index=False)
    print(f"Save predictions to {output_file}")





if __name__ == "__main__":
    # # train
    # iter_count = 3
    # train_path = '../data/error_label/iter{}/iter{}-verify.xlsx'.format(str(iter_count), str(iter_count))
    # dev_path = '../data/error_label/dev-verify.xlsx'
    # if iter_count == 0:
    #     # initialization
    #     model_name = '../model/pretrained_model/roberta-base/'
    # else:
    #     # continue training
    #     model_name = '../model/error_label/iter{}/verify/'.format(str(iter_count-1))
    # model_type = 'roberta'
    # tokenizer_path = '../model/pretrained_model/roberta-base/'
    # train_output_file = '../result/error_label/iter{}/train-verify.xlsx'.format(str(iter_count))
    # dev_output_file = '../result/error_label/iter{}/dev-verify.xlsx'.format(str(iter_count))
    # output_dir = "../model/error_label/iter{}/verify/".format(str(iter_count))
    # res_col = 'pred_error'
    # train_model(train_path, model_name, res_col, model_type, dev_path=dev_path, tokenizer_path=tokenizer_path, batch_size=32, output_dir=output_dir, output_file=train_output_file, dev_output_file=dev_output_file)


    # predict
    model_type = 'roberta'
    model_size = 'base'
    iter_count = 3
    test_path = '../data/iter{}/filter-{}-{}.xlsx'.format(str(iter_count), model_type, model_size)
    model_path = '../model/error_label/iter{}/verify/'.format(str(iter_count))
    tokenizer_path = '../model/pretrained_model/roberta-base/'
    output_file = '../data/iter{}/filter-{}-{}-verify.xlsx'.format(str(iter_count), model_type, model_size)
    res_col = 'pred_error'
    predict_and_save_texts_only(test_path, model_path, tokenizer_path, res_col, output_file, model_type)
