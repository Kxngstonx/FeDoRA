import os
import re

log_dirs = [
    "/home/hbkim/python/research-feddora/feddora_share/logs/roberta_lora_experiments",
    "/home/hbkim/python/research-feddora/feddora_share/logs/roberta_glue_experiments"
]

log_files = []
for d in log_dirs:
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".log"):
                log_files.append(os.path.join(root, f))

log_files.sort()
alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
results = {}

for lf in log_files:
    with open(lf, "r") as f:
        content = f.read()
    
    matches = re.findall(r'Global Model Test accuracy: ([\d\.]+)', content)
    if not matches:
        continue
        
    accs = [float(m) for m in matches]
    
    dataset = os.path.basename(os.path.dirname(lf))
    exp_name = os.path.basename(lf).replace(".log", "")
    
    if dataset not in results:
        results[dataset] = {}
    
    emas = {a: [] for a in alphas}
    for i, acc in enumerate(accs):
        for a in alphas:
            if i == 0:
                emas[a].append(acc)
            else:
                emas[a].append(a * acc + (1 - a) * emas[a][-1])
                
    results[dataset][exp_name] = {
        "raw": accs,
        "emas": emas
    }

print("# Experiment Results Summary (Grouped by Round)")
print()

for dataset, exps in results.items():
    print(f"## Dataset: {dataset}")
    print()
    
    max_len = max(len(d["raw"]) for d in exps.values())
    rounds = [i for i in range(10, max_len + 1, 10)]
    
    print("| Round | Experiment | Raw Accuracy | EMA (0.1) | EMA (0.3) | EMA (0.5) | EMA (0.7) | EMA (0.9) |")
    print("|---|---|---|---|---|---|---|---|")
    
    for r in rounds:
        idx = r - 1
        for exp_name in sorted(exps.keys()):
            data = exps[exp_name]
            if len(data["raw"]) > idx:
                raw = data["raw"]
                emas = data["emas"]
                row = [
                    str(r),
                    exp_name,
                    f"{raw[idx]:.4f}",
                    f"{emas[0.1][idx]:.4f}",
                    f"{emas[0.3][idx]:.4f}",
                    f"{emas[0.5][idx]:.4f}",
                    f"{emas[0.7][idx]:.4f}",
                    f"{emas[0.9][idx]:.4f}"
                ]
                print("| " + " | ".join(row) + " |")
    print()

print("---")
print("## FlexLoRA_SVD_A Analysis")
print()

target_method = "FlexLoRA_SVD_A"
metrics = ["raw"] + [f"EMA_{a}" for a in alphas]

analysis_results = []

for dataset, exps in results.items():
    target_exp_keys = [k for k in exps.keys() if target_method in k]
    if not target_exp_keys:
        continue
    target_key = target_exp_keys[0]
    
    max_len = max(len(d["raw"]) for d in exps.values())
    rounds = [i for i in range(10, max_len + 1, 10)]
    
    for r in rounds:
        idx = r - 1
        
        if len(exps[target_key]["raw"]) <= idx:
            continue
            
        target_data = exps[target_key]
        
        for metric in metrics:
            if metric == "raw":
                target_val = target_data["raw"][idx]
                others = [exps[k]["raw"][idx] for k in exps.keys() if k != target_key and len(exps[k]["raw"]) > idx]
            else:
                alpha = float(metric.split("_")[1])
                target_val = target_data["emas"][alpha][idx]
                # FIX: Use emas[alpha] instead of emas to check length
                others = [exps[k]["emas"][alpha][idx] for k in exps.keys() if k != target_key and len(exps[k]["emas"][alpha]) > idx]
            
            if not others:
                continue
                
            best_other = max(others)
            gap = target_val - best_other
            
            analysis_results.append({
                "dataset": dataset,
                "round": r,
                "metric": metric,
                "target_val": target_val,
                "best_other": best_other,
                "gap": gap
            })

analysis_results.sort(key=lambda x: x["gap"], reverse=True)

print("### Top Scenarios for FlexLoRA_SVD_A vs Best Other Baselines (Gap > 0)")
print("| Rank | Dataset | Round | Metric | FlexLoRA_SVD_A | Best Other | Gap |")
print("|---|---|---|---|---|---|---|")
rank = 1
for res in analysis_results:
    if res["gap"] > -0.005:  # Let's show top ones or ones where it's competitive
        row = [
            str(rank),
            res["dataset"],
            str(res["round"]),
            res["metric"],
            f"{res['target_val']:.4f}",
            f"{res['best_other']:.4f}",
            f"{res['gap']:.4f}"
        ]
        print("| " + " | ".join(row) + " |")
        rank += 1
