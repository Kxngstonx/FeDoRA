import os
import glob
import re

log_dir = "logs/vit_cifar100_svhn_experiments"
log_files = glob.glob(os.path.join(log_dir, "**/*.log"), recursive=True)

completed = []
in_progress = []

for filepath in log_files:
    filename = os.path.basename(filepath)
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    last_round = -1
    acc = None
    
    for line in lines:
        if "ROUND" in line:
            match = re.search(r"ROUND\s+(\d+)", line)
            if match:
                last_round = int(match.group(1))
        if "Global Model Test accuracy:" in line:
            match = re.search(r"Global Model Test accuracy:\s+([\d\.]+)", line)
            if match:
                acc = float(match.group(1))
                
    if last_round >= 49:
        completed.append((filename, acc))
    else:
        in_progress.append((filename, last_round, acc))

print(f"Total log files: {len(log_files)}")
print(f"Completed: {len(completed)}")
for f, acc in completed:
    print(f"  [DONE] {f}: Acc={acc:.4f}")

print(f"\nIn Progress: {len(in_progress)}")
for f, r, acc in in_progress:
    acc_str = f"{acc:.4f}" if acc is not None else "N/A"
    print(f"  [RUNNING] {f}: Round {r}, Latest Acc={acc_str}")
