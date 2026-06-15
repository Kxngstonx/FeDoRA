import torch
from main import init_nets
from peft_utils import inject_peft

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
    flex_lora = False
    trainable_A = False
    ravan_heads = 4
    ravan_init = 'gram_schmidt'
    use_projection_head = False

args = Args()
model = init_nets(dataset, 1, args, 'cpu', base=True)[0]

# --- FIX START ---
if args.peft != 'none':
    for param in model.parameters():
        param.requires_grad = False

_ft_classifier_global_skip = []
if args.peft != 'none' and getattr(args, 'ft_classifier', False):
    if 'roberta' in args.model:
        _ft_classifier_global_skip = ['classifier']
    elif 'qwen' in args.model:
        _ft_classifier_global_skip = ['score']
    elif 'vit' in args.model:
        _ft_classifier_global_skip = ['classifier'] # Added for ViT!

_peft_kwargs = dict(
    trainable_A=args.trainable_A, 
    global_skip_modules=_ft_classifier_global_skip, 
    target_modules=args.target_modules, 
    ravan_heads=args.ravan_heads, 
    ravan_init=args.ravan_init
)
model = inject_peft(model, args.peft, args.lora_r, args.lora_alpha, 0.0, **_peft_kwargs)
# --- FIX END ---

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable}, Total: {total}, Ratio: {trainable/total:.4f}")

for name, p in model.named_parameters():
    if p.requires_grad:
        print(f"Trainable: {name}")

