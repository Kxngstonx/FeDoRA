print("="*80)
print("논문 vs 코드 세팅 라인 바이 라인 대조")
print("="*80)

# ===== Table 7: FL Hyperparameters =====
print("\n[Table 7] FL Hyperparameters for ViT-B-16/CIFAR-100 & SVHN")
paper = {
    'batch_size': 32,
    'local_iterations': 50,
    'communication_rounds': 50,
    'epochs_per_round': 1,
    'optimizer': 'ADAM (momentum=0.9)',
    'participation': '3 clients per round (uniform random)',
}
code = {
    'batch_size': 32,
    'local_iterations': 50,  # --local_steps 50
    'communication_rounds': 50,  # --round 50
    'epochs_per_round': 1,  # --epochs 1
    'optimizer': 'AdamW',  # --optimizer adamw
    'participation_20c': '3/20 = 0.15',
    'participation_50c': '3/50 = 0.06',
}
for k, pv in paper.items():
    cv = code.get(k, 'N/A')
    match = '✅' if str(pv) == str(cv) else '⚠️'
    print(f"  {match} {k}: Paper={pv} | Code={cv}")

print()
print("  ⚠️  ISSUE 1: Paper uses 'ADAM with momentum=0.9'")
print("     Code uses 'AdamW'. AdamW != Adam. Also, Adam momentum=0.9 means beta1=0.9")
print("     PyTorch Adam default beta1=0.9, so that's fine, but AdamW adds weight decay.")
print("     → FIX: Change --optimizer adam OR ensure AdamW weight_decay=0")

# ===== Table 9: LoRA Target Modules =====
print("\n[Table 9] LoRA Target Modules")
print("  Paper: ViT-B-16 → query, value")
print("  Code:  timm ViT → qkv (fused)")
print("  ✅ FIXED: target_modules changed to 'qkv'")
print("     Note: timm ViT fuses Q,K,V into single Linear. 'qkv' is correct mapping.")

# ===== Table 10: LoRA Ranks =====
print("\n[Table 10] LoRA Ranks (Lower Budget)")
table10 = {
    'FedIT': (32, 32),
    'FedEx-LoRA': (32, 32),
    'FFA-LoRA': (64, 64),
    'RAVAN': (110, 110),
}
for method, (paper_r, code_r) in table10.items():
    match = '✅' if paper_r == code_r else '❌'
    print(f"  {match} {method}: Paper r={paper_r} | Code r={code_r}")

# ===== Table 8a: Optimal LRs (Lower Budget / IID / CIFAR-100) =====
print("\n[Table 8a] Learning Rates: Lower Budget / IID / ViT-B-16/CIFAR-100")
paper_lrs = {
    'FedIT':      '5e-3',
    'FedEx-LoRA': '1e-3',
    'FFA-LoRA':   '1e-2',  # Paper says 1e-2 for FFA-LoRA CIFAR-100 IID lower!
    'RAVAN':      '5e-4',  # Paper says 5e-4 for RAVAN CIFAR-100 IID lower!
}
code_lrs = {
    'FedIT':      '5e-3',
    'FedEx-LoRA': '1e-3',
    'FFA-LoRA':   '1e-3',  # We changed to 1e-3 
    'RAVAN':      '1e-4',  # We changed to 1e-4
}
for method in paper_lrs:
    match = '✅' if paper_lrs[method] == code_lrs[method] else '❌'
    print(f"  {match} {method}: Paper={paper_lrs[method]} | Code={code_lrs[method]}")

print()
print("  ❌ ISSUE 2: FFA-LoRA LR wrong! Paper=1e-2, Code=1e-3")
print("  ❌ ISSUE 3: RAVAN LR wrong! Paper=5e-4, Code=1e-4")
print("     → FIX: Restore FFA-LoRA to 1e-2, RAVAN to 5e-4")

# ===== Table 8a: Optimal LRs for SVHN =====
print("\n[Table 8a] Learning Rates: Lower Budget / IID / ViT-B-16/SVHN")
paper_svhn = {
    'FedIT':      '1e-3',
    'FedEx-LoRA': '1e-3',
    'FFA-LoRA':   '1e-2',
    'RAVAN':      '5e-4',
}
code_svhn = {
    'FedIT':      '1e-3',
    'FedEx-LoRA': '1e-3',
    'FFA-LoRA':   '1e-3',
    'RAVAN':      '1e-4',
}
for method in paper_svhn:
    match = '✅' if paper_svhn[method] == code_svhn[method] else '❌'
    print(f"  {match} {method}: Paper={paper_svhn[method]} | Code={code_svhn[method]}")

# ===== Table 8c: Non-IID LRs =====
print("\n[Table 8c] Learning Rates: Lower Budget / Non-IID / ViT-B-16/CIFAR-100")
paper_noniid = {
    'FedIT':      '5e-3',
    'FedEx-LoRA': '1e-3',
    'FFA-LoRA':   '1e-2',
    'RAVAN':      '5e-4',
}
code_noniid = {
    'FedIT':      '5e-3',
    'FedEx-LoRA': '1e-3',
    'FFA-LoRA':   '1e-3',
    'RAVAN':      '1e-4',
}
for method in paper_noniid:
    match = '✅' if paper_noniid[method] == code_noniid[method] else '❌'
    print(f"  {match} {method}: Paper={paper_noniid[method]} | Code={code_noniid[method]}")

print("\n" + "="*80)
print("최종 수정 항목 요약")
print("="*80)
print("1. FFA-LoRA LR: 1e-3 → 1e-2 (CIFAR-100 IID/NonIID, SVHN IID/NonIID)")
print("2. RAVAN LR: 1e-4 → 5e-4 (CIFAR-100 IID/NonIID, SVHN IID/NonIID)")
print("3. Optimizer: adamw → adam (Paper uses ADAM, not AdamW)")
print("4. FlexLoRA, FeDoRA: 논문에 없는 기법이므로 sweep 필요 (현재 5e-3 유지)")
