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
import os


# Settings
EPOCH = 30
MAX_LENGTH = 512
LABEL_LIST = ['Incomplete', 'Not A Question', 
              'Spell Error', 'Grammar Error', 'Vague', 'Unnecessary Copy from Passage', 
              'Off Topic', 'Factual Error', 'Information Not Mentioned', 'Off Target Answer',  
              'No Error']
mlb = MultiLabelBinarizer(classes=LABEL_LIST)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CustomBertForMultiLabel(AutoModelForSequenceClassification):
    def __init__(self, config, loss_type='bce', class_weights=None):
        super().__init__(config)
        if loss_type == 'focal':
            self.loss_fn = FocalLoss()
        elif loss_type == 'weighted':
            assert class_weights is not None, "Provide class_weights for weighted loss"
            weights = torch.tensor(class_weights, dtype=torch.float)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
            labels=None  
        )
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits}

def get_sep_token(model_type):
    return {
        'bert': '[SEP]',
        'deberta': '[SEP]',
        'roberta': '</s>',
    }.get(model_type.lower(), '[SEP]')

# Load Data
def load_and_preprocess_data(path, model_type='roberta'):
    df = pd.read_excel(path)
    df = df.dropna(subset=['error_label'])
    df['label_list'] = df['error_label'].apply(
        lambda x: [lbl.strip() for lbl in x.split(';') if lbl.strip() in LABEL_LIST]
    )
    labels = mlb.fit_transform(df['label_list'])
    sep_token = get_sep_token(model_type)
    # df['text'] = (
    #     "Question: " + df['question'] + sep_token +
    #     "Answer: " + df['answer'] + sep_token +
    #     "Passage: " + df['passage']
    # )
    df['text'] = df['question'] + sep_token + df['answer'] + sep_token + df['passage']
    return df, df['text'].tolist(), labels

# Custom Dataset
class ErrorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH)
        self.labels = torch.tensor(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

class TextOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

# Metrics
def compute_metrics(pred, thres=0.5):
    logits = pred.predictions
    probs = 1 / (1 + np.exp(-logits))         # sigmoid
    preds = (probs > thres).astype(int)
    labels = pred.label_ids
    return {
        'accuracy': accuracy_score(labels, preds),
        'micro_f1': f1_score(labels, preds, average='micro'),
        'macro_f1': f1_score(labels, preds, average='macro'),
        'precision': precision_score(labels, preds, average='micro'),
        'recall': recall_score(labels, preds, average='micro'),
    }

# train model
def train_model(train_path, model_name, model_type, dev_path=None, tokenizer_path=None, batch_size=8, output_dir="outputs", output_file="train_predictions.xlsx", dev_output_file='', thres=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_df, train_texts, train_labels = load_and_preprocess_data(train_path, model_type)
    if dev_path:
        dev_df, dev_texts, dev_labels = load_and_preprocess_data(dev_path, model_type)
    else:
        train_texts, dev_texts, train_labels, dev_labels, train_df, dev_df = train_test_split(
            train_texts, train_labels, train_df, 
            test_size=0.1, 
            random_state=42
        )
    print(f"Train size: {len(train_df)}, Dev size: {len(dev_df)}")
    train_dataset = ErrorDataset(train_texts, train_labels, tokenizer)
    dev_dataset = ErrorDataset(dev_texts, dev_labels, tokenizer)

    model = CustomBertForMultiLabel.from_pretrained(
        model_name, 
        num_labels=len(LABEL_LIST)
    )
    # model.loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    
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
        metric_for_best_model="micro_f1",
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

    # predict train set
    print('******** Start Train Predicting ********')
    predict_and_save(trainer, train_df, tokenizer, train_texts, train_labels, thres=thres, output_file=output_file)
    # predict dev set
    if dev_output_file:
        print('******** Start Dev Predicting ********')
        predict_and_save(trainer, dev_df, tokenizer, dev_texts, dev_labels, thres=thres, output_file=dev_output_file)

# predict and save the result with metrics
def predict_and_save(trainer, dev_df, tokenizer, dev_texts, dev_labels, thres=0.5, output_file="dev_predictions.xlsx"):
    pred_output = trainer.predict(ErrorDataset(dev_texts, dev_labels, tokenizer))
    logits = pred_output.predictions
    probs = 1 / (1 + np.exp(-logits))  
    pred_binary = (probs > thres).astype(int)
    pred_labels = mlb.inverse_transform(pred_binary)
    dev_df['predicted_label'] = ['; '.join(lbls) if lbls else '' for lbls in pred_labels]
    output_dir = os.path.dirname(output_file)
    ensure_directory_exists(output_dir)
    dev_df.to_excel(output_file, index=False)
    print(f"Save result to {output_file}")

    print("Classification Report:")
    print("  Micro F1:     ", f1_score(dev_labels, pred_binary, average='micro'))
    print("  Macro F1:     ", f1_score(dev_labels, pred_binary, average='macro'))
    print("  Micro P/R:    ", precision_score(dev_labels, pred_binary, average='micro'),
                                  "/", recall_score(dev_labels, pred_binary, average='micro'))
    print("  Accuracy:     ", accuracy_score(dev_labels, pred_binary))

    report = classification_report(dev_labels, pred_binary, target_names=mlb.classes_)
    print(report)

# predict and save prediction only
def predict_and_save_texts_only(
    df, model_path, tokenizer_path, texts, mlb, thresholds=None, 
    output_file="predictions.xlsx", col_name='pred_error', need_prob=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    dataset = TextOnlyDataset(texts, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    all_logits = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.cpu().numpy()
            all_logits.append(logits)
    logits = np.concatenate(all_logits, axis=0)

    probs = 1 / (1 + np.exp(-logits)) 

    num_labels = probs.shape[1]
    if isinstance(thresholds, float):
        thresholds = np.array([thresholds] * num_labels)
    elif isinstance(thresholds, list):
        thresholds = np.array(thresholds)
    elif thresholds is None:
        thresholds = np.array([0.5] * num_labels)

    pred_binary = (probs > thresholds).astype(int)
    pred_labels = mlb.inverse_transform(pred_binary)

    # label convert
    df[col_name] = ['; '.join(lbls) if lbls else '' for lbls in pred_labels]

    # prob for each error type
    if need_prob:
        for i, label in enumerate(mlb.classes_):
            df[f"{label}_prob"] = probs[:, i]

    output_dir = os.path.dirname(output_file)
    ensure_directory_exists(output_dir)
    df.to_excel(output_file, index=False)
    print(f"Save predictions to {output_file}")



if __name__ == "__main__":
    # # train
    # iter_count = 3
    # train_path = '../data/error_label/iter{}/iter{}-train.xlsx'.format(str(iter_count), str(iter_count))
    # dev_path = '../data/error_label/dev.xlsx'
    # model_type = 'roberta'
    # model_size = 'base'
    # if iter_count == 0:
    #     model_name = '../model/pretrained_model/{}-{}/'.format(model_type, model_size)
    # else:
    #     model_name = '../model/error_label/iter{}/{}-{}'.format(str(iter_count-1), model_type, model_size)
    # tokenizer_path = '../model/pretrained_model/{}-{}/'.format(model_type, model_size)
    # train_output_file = '../result/error_label/iter{}/train-pred-{}-{}.xlsx'.format(str(iter_count), model_type, model_size)
    # dev_output_file = '../result/error_label/iter{}/dev-pred-{}-{}.xlsx'.format(str(iter_count), model_type, model_size)
    # output_dir = "../model/error_label/iter{}/{}-{}".format(str(iter_count), model_type, model_size)
    # train_model(train_path, model_name, model_type, dev_path=dev_path,tokenizer_path=tokenizer_path, batch_size=32, output_dir=output_dir, output_file=train_output_file, dev_output_file=dev_output_file, thres=0.5)

    # predict
    df = pd.read_excel('../data/test/qgeval.xlsx')
    model_type = 'roberta'
    model_size = 'base'
    iter_count = 3
    model_path = "../model/error_label/iter{}/{}-{}/".format(str(iter_count), model_type, model_size)
    tokenizer_path="../model/pretrained_model/{}-{}/".format(model_type, model_size)
    sep_token = get_sep_token(model_type)
    texts = [
        f"{row['question']}{sep_token}{row['answer']}{sep_token}{row['passage']}"
        for _, row in df.iterrows()
    ]
    thresholds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    output_file = '../result/test/iter{}/{}-{}/qgeval-el.xlsx'.format(str(iter_count), model_type, model_size)
    mlb.fit([[label] for label in LABEL_LIST])
    predict_and_save_texts_only(df, model_path, tokenizer_path, texts, mlb, thresholds=thresholds, output_file=output_file, col_name='pred_error', need_prob=False)