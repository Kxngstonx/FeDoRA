import os
import sys
import csv
import subprocess
import textwrap
import torch
from torch.utils.data import Dataset, ConcatDataset
from transformers import AutoTokenizer


_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'anli')

# ANLI has 3 rounds: R1, R2, R3
# Labels: 0=entailment, 1=neutral, 2=contradiction
ANLI_ROUNDS = ['R1', 'R2', 'R3']
ANLI_FIELDS = ['premise', 'hypothesis']
ANLI_NUM_CLASSES = 3


def _csv_path(round_name, split):
    """e.g., data/anli/R1/train.csv"""
    return os.path.join(_CACHE_DIR, round_name, f'{split}.csv')


def _download_split(round_name, split_name, out_path):
    """Download one ANLI split via a subprocess to avoid local 'datasets/' shadowing HF library."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # HF datasets split naming: train_r1, dev_r1, test_r1, etc.
    hf_split = f'{split_name}_{round_name.lower()}'
    script = textwrap.dedent(f"""
        import sys
        sys.path = [p for p in sys.path if p != '' and '/fedora' not in p]
        from datasets import load_dataset
        import csv, os

        raw = load_dataset("anli")
        split = raw[{repr(hf_split)}]
        out = {repr(out_path)}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in split:
                if item['label'] == -1:
                    continue
                writer.writerow([item['premise'], item['hypothesis'], item['label']])
        print(f"Saved {{len(split)}} rows -> {{out}}")
    """)
    print(f"Downloading ANLI {round_name}/{split_name} ...")
    subprocess.run([sys.executable, '-c', script], check=True)


def _ensure_cached(round_name):
    for split in ('train', 'test'):
        path = _csv_path(round_name, split)
        if not os.path.exists(path):
            _download_split(round_name, split, path)


class ANLIRoundDataset(Dataset):
    """Single-round ANLI dataset (e.g., R1 train or R2 test)."""

    def __init__(self, round_name, split='train', model_name="roberta-large", max_length=128):
        assert round_name in ANLI_ROUNDS, f"round_name must be one of {ANLI_ROUNDS}"
        assert split in ('train', 'test'), f"split must be 'train' or 'test'"

        self.round_name = round_name
        self.num_classes = ANLI_NUM_CLASSES
        self.max_length = max_length

        _ensure_cached(round_name)
        path = _csv_path(round_name, split)

        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                premise = row[0]
                hypothesis = row[1]
                label = int(row[2])
                self.data.append(([premise, hypothesis], label))

        self.targets = [item[1] for item in self.data]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        texts, label = self.data[idx]

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


class ANLICombinedTrainDataset(Dataset):
    """Combined R1+R2+R3 train dataset for ANLI."""

    def __init__(self, model_name="roberta-large", max_length=128):
        self.num_classes = ANLI_NUM_CLASSES
        self.max_length = max_length

        # Load all three rounds
        self._round_datasets = []
        for rnd in ANLI_ROUNDS:
            ds = ANLIRoundDataset(rnd, split='train', model_name=model_name, max_length=max_length)
            self._round_datasets.append(ds)

        # Flatten all data and targets
        self.data = []
        self.targets = []
        for ds in self._round_datasets:
            self.data.extend(ds.data)
            self.targets.extend(ds.targets)

        # Reuse tokenizer from first round dataset
        self.tokenizer = self._round_datasets[0].tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        texts, label = self.data[idx]

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
