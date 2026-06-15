import glob
import os
import re

results = {}
for filepath in glob.glob("./logs/vit_cifar100_svhn_experiments/**/*.log", recursive=True):
    filename = os.path.basename(filepath)
    if 'seed' not in filename: continue
    
    parts = filename.replace('.log', '').split('_')
    if len(parts) >= 6:
        dataset = parts[0]
        partition = parts[2]
        method = parts[4]
        
        max_acc = 0.0
        with open(filepath, 'r') as f:
            for line in f:
                if 'Global Model Test accuracy:' in line:
                    match = re.search(r'Global Model Test accuracy:\s*([0-9.]+)', line)
                    if match:
                        acc = float(match.group(1))
                        if acc > max_acc:
                            max_acc = acc
        
        if method not in results:
            results[method] = {}
        key = f"{dataset} ({partition.upper()})"
        results[method][key] = max_acc

for method, perfs in results.items():
    print(f"{method}: {perfs}")
