import os
import sys
import csv
import subprocess
import textwrap
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

_CACHE_DIR = '/mnt/data1/hbkim/mmlu_data'


def _csv_path(split):
    return os.path.join(_CACHE_DIR, f'{split}.csv')


def _download_split(split_name, out_path):
    """Download one MMLU split via a subprocess to avoid local 'datasets/' shadowing HF library."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # We map 'train' -> 'auxiliary_train', 'validation' -> 'test' as per MMLU typical setup
    actual_split = 'auxiliary_train' if split_name == 'train' else 'test'
    
    script = textwrap.dedent(f"""
        import sys
        import os
        # Use NVMe SSD for HF cache
        os.environ['HF_DATASETS_CACHE'] = '/mnt/data1/hbkim/hf_cache'
        os.environ['HF_HOME'] = '/mnt/data1/hbkim/hf_cache'
        
        # Strip the project root so the local datasets/ package doesn't shadow HF datasets.
        sys.path = [p for p in sys.path if p != '' and '/feddora' not in p]
        from datasets import load_dataset
        import csv

        raw = load_dataset("cais/mmlu", "all", split={repr(actual_split)})
        out = {repr(out_path)}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in raw:
                question = item['question']
                choices = item['choices']
                answer = item['answer']
                
                # Combine question and choices into a single string
                text = f"{{question}} \\n (A) {{choices[0]}} \\n (B) {{choices[1]}} \\n (C) {{choices[2]}} \\n (D) {{choices[3]}}"
                
                writer.writerow([text, answer])
                
        print(f"Saved MMLU {{out}}")
    """)
    print(f"Downloading MMLU {split_name} (using {actual_split}) ...")
    subprocess.run([sys.executable, '-c', script], check=True)


def _ensure_cached():
    for split in ('train', 'validation'):
        path = _csv_path(split)
        if not os.path.exists(path):
            _download_split(split, path)


class MMLUDataset(Dataset):
    def __init__(self, train=True, model_name="roberta-large", max_length=512):
        self.num_classes = 4
        self.max_length = max_length

        _ensure_cached()

        split = 'train' if train else 'validation'
        path = _csv_path(split)

        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                text = row[0]
                label = int(row[1])
                self.data.append((text, label))

        self.targets = [item[1] for item in self.data]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }, label
