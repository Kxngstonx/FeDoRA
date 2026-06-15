import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

class RoBERTaLargeWrapper(nn.Module):
    def __init__(self, num_classes=4, model_name="roberta-large"):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_classes

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True)

        # Re-initialize classifier head explicitly (dense + out_proj)
        with torch.no_grad():
            nn.init.normal_(self.model.classifier.dense.weight, mean=0.0, std=config.initializer_range)
            nn.init.zeros_(self.model.classifier.dense.bias)
            nn.init.normal_(self.model.classifier.out_proj.weight, mean=0.0, std=config.initializer_range)
            nn.init.zeros_(self.model.classifier.out_proj.bias)

        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()

    def forward(self, x):
        if isinstance(x, dict):
            outputs = self.model(
                input_ids=x['input_ids'],
                attention_mask=x['attention_mask'],
                output_hidden_states=True,
            )
        else:
            outputs = self.model(x, output_hidden_states=True)

        logits = outputs.logits
        # CLS token from last hidden state — consistent with RoBERTa's internal pooler
        pooled = outputs.hidden_states[-1][:, 0, :]
        return pooled, logits
