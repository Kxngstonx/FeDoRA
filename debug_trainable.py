import sys, torch
sys.path.insert(0, '.')
from utils import init_nets
from peft_utils import inject_peft

class DummyDataset:
    num_classes = 100

dataset = DummyDataset()

class Args:
    model = 'vit-base'
    group_norm = False
    in_channels = 3
    dataset = 'cifar100'
    use_projection_head = False

args = Args()

# Test each method
configs = {
    'FedIT': {'peft': 'lora', 'trainable_A': True, 'r': 32, 'alpha': 32, 'target': ['query','value'], 'flex_lora': False, 'flex_lora_freeze_a': False, 'flex_lora_svd_a': False, 'fedex_lora': False},
    'FedEx-LoRA': {'peft': 'lora', 'trainable_A': True, 'r': 32, 'alpha': 32, 'target': ['query','value'], 'flex_lora': False, 'flex_lora_freeze_a': False, 'flex_lora_svd_a': False, 'fedex_lora': True},
    'FFA-LoRA': {'peft': 'lora', 'trainable_A': False, 'r': 64, 'alpha': 64, 'target': ['query','value'], 'flex_lora': True, 'flex_lora_freeze_a': True, 'flex_lora_svd_a': False, 'fedex_lora': False},
    'FlexLoRA': {'peft': 'lora', 'trainable_A': True, 'r': 32, 'alpha': 32, 'target': ['query','value'], 'flex_lora': True, 'flex_lora_freeze_a': False, 'flex_lora_svd_a': False, 'fedex_lora': False},
    'FeDoRA': {'peft': 'dora', 'trainable_A': True, 'r': 32, 'alpha': 32, 'target': ['query','value'], 'flex_lora': True, 'flex_lora_freeze_a': False, 'flex_lora_svd_a': True, 'fedex_lora': False},
    'RAVAN': {'peft': 'ravan', 'trainable_A': False, 'r': 110, 'alpha': 110, 'target': ['query','value'], 'flex_lora': False, 'flex_lora_freeze_a': False, 'flex_lora_svd_a': False, 'fedex_lora': False},
}

for method_name, cfg in configs.items():
    model = init_nets(dataset, 1, args, 'cpu', base=True)[0]
    
    # Freeze base model first (the fix we applied)
    for param in model.parameters():
        param.requires_grad = False
    
    global_skip = ['classifier']  # ViT classifier
    
    peft_kwargs = dict(
        trainable_A=cfg['trainable_A'],
        global_skip_modules=global_skip,
        target_modules=cfg['target'],
        ravan_heads=4,
        ravan_init='gram_schmidt'
    )
    model = inject_peft(model, cfg['peft'], cfg['r'], cfg['alpha'], 0.0, **peft_kwargs)
    
    trainable_params = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable_params[name] = p.numel()
    
    total_trainable = sum(trainable_params.values())
    lora_params = sum(v for k,v in trainable_params.items() if 'lora' in k or 'ravan' in k or 'dora' in k or '.m' in k)
    classifier_params = sum(v for k,v in trainable_params.items() if 'classifier' in k)
    
    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"  Total trainable: {total_trainable:,}")
    print(f"  LoRA/PEFT params: {lora_params:,}")
    print(f"  Classifier params: {classifier_params:,}")
    print(f"  Other params: {total_trainable - lora_params - classifier_params:,}")
    
    # Show first few trainable param names
    for name in list(trainable_params.keys())[:5]:
        print(f"    - {name}: {trainable_params[name]:,}")
    if len(trainable_params) > 5:
        print(f"    ... and {len(trainable_params)-5} more")
