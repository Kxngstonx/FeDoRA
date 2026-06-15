import os
import sys
import csv
import subprocess
import textwrap
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

SUPERGLUE_TASK_INFO = {
    'boolq': {'fields': ['question', 'passage'], 'num_classes': 2, 'val_split': 'validation'},
    'multirc': {'fields': ['paragraph', 'question_answer'], 'num_classes': 2, 'val_split': 'validation'},
    'wic': {'fields': ['word_sentence1', 'sentence2'], 'num_classes': 2, 'val_split': 'validation'},
    'record': {'fields': ['passage', 'query_with_entity'], 'num_classes': 2, 'val_split': 'validation'}
}

_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'superglue')


def _csv_path(task, split):
    return os.path.join(_CACHE_DIR, task, f'{split}.csv')


def _download_split(task, split_name, out_path):
    """Download one SuperGLUE split via a subprocess to avoid local 'datasets/' shadowing HF library."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    script = textwrap.dedent(f"""
        import sys
        # Strip the project root so the local datasets/ package doesn't shadow HF datasets.
        sys.path = [p for p in sys.path if p != '' and '/feddora' not in p]
        from datasets import load_dataset
        import csv, os

        raw = load_dataset("super_glue", {repr(task)})
        split = raw[{repr(split_name)}]
        out = {repr(out_path)}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in split:
                if {repr(task)} == 'boolq':
                    if item['label'] == -1: continue
                    writer.writerow([item['question'], item['passage'], item['label']])
                elif {repr(task)} == 'multirc':
                    if item['label'] == -1: continue
                    # Concatenate question and answer
                    qa = str(item['question']) + " " + str(item['answer'])
                    writer.writerow([item['paragraph'], qa, item['label']])
                elif {repr(task)} == 'wic':
                    if item['label'] == -1: continue
                    # Concatenate word and sentence1
                    word_s1 = str(item['word']) + ": " + str(item['sentence1'])
                    writer.writerow([word_s1, item['sentence2'], item['label']])
                elif {repr(task)} == 'record':
                    # Unfold ReCoRD: for each entity, replace @placeholder in query
                    passage = item['passage']
                    query = item['query']
                    answers = set(item['answers']) if item['answers'] else set()
                    entities = set(item['entities']) if item['entities'] else set()
                    for ent in entities:
                        # Label is 1 if entity is among the answers
                        label = 1 if ent in answers else 0
                        # Replace placeholder with the entity
                        q_with_ent = query.replace('@placeholder', ent)
                        writer.writerow([passage, q_with_ent, label])
        print(f"Saved rows -> {{out}}")
    """)
    print(f"Downloading SuperGLUE {task}/{split_name} ...")
    subprocess.run([sys.executable, '-c', script], check=True)


def _ensure_cached(task):
    info = SUPERGLUE_TASK_INFO[task]
    for split in ('train', info['val_split']):
        path = _csv_path(task, split)
        if not os.path.exists(path):
            _download_split(task, split, path)


class SuperGLUEDataset(Dataset):
    def __init__(self, task, train=True, model_name="roberta-large", max_length=128):
        if task not in SUPERGLUE_TASK_INFO:
            raise ValueError(f"Unsupported SuperGLUE task: '{task}'. Choose from {list(SUPERGLUE_TASK_INFO)}")

        info = SUPERGLUE_TASK_INFO[task]
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
