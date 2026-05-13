import torch
from torch.utils.data import Dataset
import os
import csv
import urllib.request
from transformers import AutoTokenizer

_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'ag_news')

class AGNewsDataset(Dataset):
    def __init__(self, train=True, model_name="Qwen/Qwen2.5-0.5B", max_length=128):
        self.max_length = max_length
        self.num_classes = 4

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        os.makedirs(_CACHE_DIR, exist_ok=True)
        if train:
            url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
            filename = os.path.join(_CACHE_DIR, 'train.csv')
        else:
            url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
            filename = os.path.join(_CACHE_DIR, 'test.csv')

        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)

        self.data = []
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # AG News CSV format: Class Index (1-4), Title, Description
                label = int(row[0]) - 1 # 0 to 3
                text = row[1] + " " + row[2]
                self.data.append((text, label))
                
        # For memory efficiency, pre-tokenize or just tokenize on the fly.
        # We will tokenize on the fly in __getitem__ to save RAM, 
        # or we could pre-tokenize if we had more time.
        self.targets = [item[1] for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        inputs = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
        
        return inputs, label
