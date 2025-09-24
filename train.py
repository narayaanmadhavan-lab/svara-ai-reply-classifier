import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

DATA_PATH = "data/replies.csv"
SEED = 42
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL_MAP = {v:k for k,v in LABEL_MAP.items()}
os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['reply']).reset_index(drop=True)
df['label_str'] = df['label'].str.lower().map(lambda x: x.strip())
df = df[df['label_str'].isin(LABEL_MAP.keys())].copy()
df['label_encoded'] = df['label_str'].map(LABEL_MAP)

print("Class distribution:\n", df['label_str'].value_counts())

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=SEED)
print("train size:", len(train_df), "val size:", len(val_df))

tf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train = tf.fit_transform(train_df['reply'])
X_val = tf.transform(val_df['reply'])

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
lr.fit(X_train, train_df['label_encoded'])
pred_val = lr.predict(X_val)
acc = accuracy_score(val_df['label_encoded'], pred_val)
f1 = f1_score(val_df['label_encoded'], pred_val, average='weighted')
print("LogReg val accuracy:", acc)
print("LogReg val weighted F1:", f1)
print(classification_report(val_df['label_encoded'], pred_val, target_names=[INV_LABEL_MAP[i] for i in sorted(INV_LABEL_MAP)]))

joblib.dump(tf, "models/tfidf_vectorizer.joblib")
joblib.dump(lr, "models/logreg_model.joblib")

lgbm = lgb.LGBMClassifier(**{'objective':'multiclass', 'num_class':3, 'random_state':SEED, 'n_estimators':200})
lgbm.fit(X_train, train_df['label_encoded'])
pred_val_lgb = lgbm.predict(X_val)
acc_lgb = accuracy_score(val_df['label_encoded'], pred_val_lgb)
f1_lgb = f1_score(val_df['label_encoded'], pred_val_lgb, average='weighted')
print("LightGBM val accuracy:", acc_lgb)
print("LightGBM val weighted F1:", f1_lgb)
print(classification_report(val_df['label_encoded'], pred_val_lgb, target_names=[INV_LABEL_MAP[i] for i in sorted(INV_LABEL_MAP)]))
joblib.dump(lgbm, "models/lgbm_model.joblib")

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(examples["reply"], truncation=True, padding="max_length", max_length=128)

hf_train = Dataset.from_pandas(train_df[['reply','label_encoded']].rename(columns={'label_encoded': 'label'}))
hf_val = Dataset.from_pandas(val_df[['reply','label_encoded']].rename(columns={'label_encoded': 'label'}))
hf_train = hf_train.map(tokenize_fn, batched=True)
hf_val = hf_val.map(tokenize_fn, batched=True)
hf_train.set_format(type='torch', columns=['input_ids','attention_mask','label'])
hf_val.set_format(type='torch', columns=['input_ids','attention_mask','label'])

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1w = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1_weighted': f1w}

training_args = TrainingArguments(
    output_dir="models/distilbert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    seed=SEED,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("models/distilbert-best")
tokenizer.save_pretrained("models/distilbert-best")

eval_res = trainer.evaluate()
print("DistilBERT val metrics:", eval_res)
