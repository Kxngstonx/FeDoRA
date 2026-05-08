import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

GLUE_TASK_INFO = {
    'sst2': {'fields': ['sentence'],              'num_classes': 2, 'val': 'validation'},
    'mnli': {'fields': ['premise', 'hypothesis'], 'num_classes': 3, 'val': 'validation_matched'},
    'qnli': {'fields': ['question', 'sentence'],  'num_classes': 2, 'val': 'validation'},
    'qqp':  {'fields': ['question1', 'question2'],'num_classes': 2, 'val': 'validation'},
}

class GLUEDataset(Dataset):
    def __init__(self, task, train=True, model_name="roberta-large", max_length=128):
        if task not in GLUE_TASK_INFO:
            raise ValueError(f"Unsupported GLUE task: '{task}'. Choose from {list(GLUE_TASK_INFO)}")

        info = GLUE_TASK_INFO[task]
        self.fields = info['fields']
        self.num_classes = info['num_classes']
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        raw = load_dataset("glue", task)
        split = 'train' if train else info['val']
        # Filter out unlabeled samples (label == -1 appears in some GLUE test splits)
        self.data = [item for item in raw[split] if item['label'] != -1]
        self.targets = [item['label'] for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        fields = self.fields

        if len(fields) == 1:
            encoded = self.tokenizer(
                item[fields[0]],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
            )
        else:
            encoded = self.tokenizer(
                item[fields[0]],
                item[fields[1]],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
            )

        inputs = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }
        return inputs, item['label']
