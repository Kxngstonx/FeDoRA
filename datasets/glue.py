import os
import sys
import csv
import subprocess
import textwrap
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

GLUE_TASK_INFO = {
    'sst2': {'fields': ['sentence'],               'num_classes': 2, 'val_split': 'validation'},
    'mnli': {'fields': ['premise', 'hypothesis'],  'num_classes': 3, 'val_split': 'validation_matched'},
    'qnli': {'fields': ['question', 'sentence'],   'num_classes': 2, 'val_split': 'validation'},
    'qqp':  {'fields': ['question1', 'question2'], 'num_classes': 2, 'val_split': 'validation'},
}

_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'glue')


def _csv_path(task, split):
    return os.path.join(_CACHE_DIR, task, f'{split}.csv')


def _download_split(task, split_name, fields, out_path):
    """Download one GLUE split via a subprocess to avoid local 'datasets/' shadowing HF library."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    script = textwrap.dedent(f"""
        import sys
        # Strip the project root so the local datasets/ package doesn't shadow HF datasets.
        sys.path = [p for p in sys.path if p != '' and '/fedora' not in p]
        from datasets import load_dataset
        import csv, os

        raw = load_dataset("glue", {repr(task)})
        split = raw[{repr(split_name)}]
        fields = {repr(fields)}
        out = {repr(out_path)}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in split:
                if item['label'] == -1:   # skip unlabelled test rows
                    continue
                writer.writerow([item[field] for field in fields] + [item['label']])
        print(f"Saved {{len(split)}} rows -> {{out}}")
    """)
    print(f"Downloading GLUE {task}/{split_name} ...")
    subprocess.run([sys.executable, '-c', script], check=True)


def _ensure_cached(task):
    info = GLUE_TASK_INFO[task]
    for split in ('train', info['val_split']):
        path = _csv_path(task, split)
        if not os.path.exists(path):
            _download_split(task, split, info['fields'], path)


class GLUEDataset(Dataset):
    def __init__(self, task, train=True, model_name="roberta-large", max_length=128):
        if task not in GLUE_TASK_INFO:
            raise ValueError(f"Unsupported GLUE task: '{task}'. Choose from {list(GLUE_TASK_INFO)}")

        info = GLUE_TASK_INFO[task]
        self.fields = info['fields']
        self.num_classes = info['num_classes']
        self.max_length = max_length

        _ensure_cached(task)

        split = 'train' if train else info['val_split']
        path = _csv_path(task, split)

        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                texts = row[:len(self.fields)]
                label = int(row[len(self.fields)])
                self.data.append((texts, label))

        self.targets = [item[1] for item in self.data]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        texts, label = self.data[idx]

        if len(texts) == 1:
            encoded = self.tokenizer(
                texts[0],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
            )
        else:
            encoded = self.tokenizer(
                texts[0],
                texts[1],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
            )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }, label
