import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

class QwenLLMWrapper(nn.Module):
    def __init__(self, num_classes=4, model_name="Qwen/Qwen2.5-0.5B"):
        super().__init__()
        
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        config.pad_token_id = config.eos_token_id # Qwen usually uses eos as pad
        # Use Sequence Classification model which adds a classification head
        self.qwen = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        # from_pretrained with ignore_mismatched_sizes=True can leave score weight
        # uninitialized (NaN) in bfloat16 on GPU; re-initialize explicitly.
        with torch.no_grad():
            nn.init.normal_(self.qwen.score.weight, mean=0.0, std=config.initializer_range)
        # Ensure gradients are computable on frozen layers if peft adds adapters
        if hasattr(self.qwen, 'enable_input_require_grads'):
            self.qwen.enable_input_require_grads()
            
    def forward(self, x):
        if isinstance(x, dict):
            outputs = self.qwen(input_ids=x['input_ids'], attention_mask=x['attention_mask'], output_hidden_states=True)
        else:
            outputs = self.qwen(x, output_hidden_states=True)
            
        logits = outputs.logits
        # Apply final RMSNorm for consistency with logits (SequenceClassifierOutputWithPast
        # carries pre-norm hidden_states; logits use post-norm last_hidden_state internally)
        pooled = self.qwen.model.norm(outputs.hidden_states[-1]).mean(dim=1)
        return pooled, logits
