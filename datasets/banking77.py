import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

class Banking77Dataset(Dataset):
    def __init__(self, train=True, model_name="roberta-base", max_length=128):
        self.max_length = max_length
        self.num_classes = 77
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        dataset = load_dataset("banking77")
        
        if train:
            self.data = dataset['train']
        else:
            self.data = dataset['test']
            
        self.targets = [item['label'] for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
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
