import torch
import torch.nn as nn
import timm

class ViTWrapper(nn.Module):
    def __init__(self, num_classes=10, model_name='vit_small_patch16_224'):
        super().__init__()
        # Use timm to load a pretrained ViT. It's lightweight and works well.
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0) # num_classes=0 removes the head
        self.classifier = nn.Linear(self.vit.num_features, num_classes)
        
        # CIFAR images are 32x32. We can interpolate the patch embeddings to support 32x32, 
        # but to keep it simple, we just use a torchvision resize transform in the dataset,
        # or we dynamically resize the batch here.
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Resize from 32x32 to 224x224
        if x.shape[-1] != 224:
            x = self.resize(x)
            
        features = self.vit(x) # (B, num_features)
        logits = self.classifier(features)
        return features, logits
