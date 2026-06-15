import glob
import os
import re

results = {}
total_experiments = 48
completed_experiments = 0

log_files = glob.glob("./logs/vit_cifar100_svhn_experiments/**/*.log", recursive=True)

for filepath in log_files:
    filename = os.path.basename(filepath)
    if 'seed' not in filename: continue
    
    parts = filename.replace('.log', '').split('_')
    if len(parts) >= 6:
        dataset = parts[0]
        clients = parts[1]
        partition = parts[2]
        method = parts[4]
        
        max_acc = 0.0
        finished = False
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Global Model Test accuracy:' in line:
                    match = re.search(r'Global Model Test accuracy:\s*([0-9.]+)', line)
                    if match:
                        acc = float(match.group(1))
                        if acc > max_acc:
                            max_acc = acc
                if 'Round 49 completed' in line or 'round:49' in line or len(lines) > 500:
                    pass
        
        # A simple way to check if it's finished is if it has 50 test accuracy logs.
        # Let's count them
        test_acc_count = sum(1 for line in lines if 'Global Model Test accuracy:' in line)
        if test_acc_count >= 50:
            finished = True
            completed_experiments += 1
            
        key = f"{dataset}_{clients}_{partition}"
        if method not in results:
            results[method] = {}
        results[method][key] = (max_acc, finished)

print(f"Completed: {completed_experiments}/{total_experiments}")
for method, perfs in results.items():
    print(f"{method}: {perfs}")
