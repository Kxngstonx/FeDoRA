import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

class HFModelWrapper(nn.Module):
    def __init__(self, num_classes, model_name="roberta-base"):
        super().__init__()
        
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        
        # padding token setting for some models that don't have it by default
        if model_name.startswith('distilbert') or model_name.startswith('roberta'):
            # roberta and distilbert usually have pad token set, but just in case
            pass
            
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()
            
    def forward(self, x):
        if isinstance(x, dict):
            outputs = self.model(input_ids=x['input_ids'], attention_mask=x['attention_mask'], output_hidden_states=True)
        else:
            outputs = self.model(x, output_hidden_states=True)
            
        logits = outputs.logits
        # For pooling, taking the mean of the last hidden state
        # In roberta, usually outputs.hidden_states[-1][:, 0, :] is used for CLS, 
        # but mean pooling works generally well too.
        hidden_states = outputs.hidden_states[-1]
        pooled = hidden_states.mean(dim=1)
        return pooled, logits
