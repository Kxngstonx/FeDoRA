import torch
import copy
from main import get_args, init_nets
from peft_utils import inject_peft
from datasets.cifar100 import CIFAR100_truncated

class DummyDataset:
    num_classes = 100

dataset = DummyDataset()
class Args:
    model = 'vit-base'
    group_norm = False
    in_channels = 3
    dataset = 'cifar100'
    peft = 'lora'
    lora_r = 32
    lora_alpha = 32
    target_modules = ['query', 'value']
    ft_classifier = True

args = Args()
model = init_nets(dataset, 1, args, 'cpu', base=True)[0]
model = inject_peft(model, args.peft, args.lora_r, args.lora_alpha, target_modules=args.target_modules)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable}, Total: {total}, Ratio: {trainable/total:.4f}")

# Check specific layers
for name, p in model.named_parameters():
    if not p.requires_grad: continue
    if 'lora' not in name:
        print(f"Trainable base param: {name}")
        break

