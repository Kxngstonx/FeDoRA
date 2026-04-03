import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoTokenizer

class Newsgroups20Dataset(Dataset):
    def __init__(self, train=True, model_name="roberta-base", max_length=128):
        self.max_length = max_length
        self.num_classes = 20
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        subset = 'train' if train else 'test'
        # remove headers, footers, and quotes to prevent overfitting on metadata
        dataset = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
        
        self.texts = dataset.data
        self.targets = dataset.target.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.targets[idx]
        
        # Handle empty strings
        if not text or not text.strip():
            text = "empty"
            
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
