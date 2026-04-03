import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

class QwenVisionWrapper(nn.Module):
    def __init__(self, num_classes=10, model_name="Qwen/Qwen2.5-0.5B"):
        super().__init__()
        
        # Patch embedding: 3 channels, 32x32 image -> 16x16 patches of size 2x2
        # For CIFAR10, image is 32x32. Patch size 4x4 -> 8x8 = 64 tokens.
        patch_size = 4
        hidden_size = 896 # Qwen2.5-0.5B hidden size
        
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        # Load Qwen model
        config = AutoConfig.from_pretrained(model_name)
        config.num_hidden_layers = 4 # Reduce layers for fast testing and lower memory
        self.qwen = AutoModelForCausalLM.from_config(config).model
        
        # Classifier head
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = self.patch_embed(x) # (B, hidden_size, 8, 8)
        x = x.flatten(2).transpose(1, 2) # (B, 64, hidden_size)
        
        # Pass through Qwen
        outputs = self.qwen(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state # (B, 64, hidden_size)
        
        # Average pooling
        pooled = hidden_states.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        return pooled, logits
