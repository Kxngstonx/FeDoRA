import os
import glob
import re

def calculate_ema(values, alpha=0.3):
    if not values: return 0.0
    ema = values[0]
    for v in values[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema

results = {}
log_files = glob.glob("logs/vit_cifar100_svhn_experiments/**/*.log", recursive=True)

for log_path in log_files:
    filename = os.path.basename(log_path)
    match = re.search(r'(cifar100|svhn)_(20c|50c)_(iid|noniid)_lower_(.*)_seed42\.log', filename)
    if not match: continue
    dataset = match.group(1)
    clients = match.group(2)
    partition = match.group(3)
    method = match.group(4)
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    is_finished = ("ROUND 49/50" in content or "ROUND 50/50" in content or "close.complete" in content)
    
    test_accs = re.findall(r'Global Model Test accuracy:\s*([0-9.]+)', content)
    if not test_accs:
        test_accs = re.findall(r'MEAN_LOCAL_ACC\s*=\s*([0-9.]+)', content)
    
    test_accs = [float(x) for x in test_accs]
    val = None
    
    if test_accs:
        val = calculate_ema(test_accs, alpha=0.3)
    
    if not is_finished and test_accs:
        val = test_accs[-1]
    
    if val is not None:
        if clients not in results: results[clients] = {}
        if method not in results[clients]: results[clients][method] = {}
        
        dataset_str = "CIFAR-100" if dataset == "cifar100" else dataset.upper()
        col_key = f"{dataset_str} ({'IID' if partition=='iid' else 'Non-IID'})"
        results[clients][method][col_key] = (val, is_finished)

method_map = {
    "FeDoRA": "Ours (FeDoRA)",
    "FedEx-LoRA": "FedEx-LoRA",
    "FedIT": "FedIT",
    "FlexLoRA": "FlexLoRA",
    "FFA-LoRA": "FFA-LoRA",
    "RAVAN": "RAVAN"
}

with open("final_performance_table_ema03_with_superglue.md", "r") as f:
    lines = f.readlines()

new_lines = []
in_table = False
current_clients = None
columns = []

for line in lines:
    if "### 20 Clients Setting" in line:
        current_clients = "20c"
    elif "### 50 Clients Setting" in line:
        current_clients = "50c"
        
    if line.strip().startswith("| Method"):
        in_table = True
        columns = [c.strip() for c in line.strip().split('|')[1:-1]]
        new_lines.append(line)
        continue
        
    if in_table and line.strip().startswith("|---"):
        new_lines.append(line)
        continue
        
    if in_table and line.strip().startswith("|"):
        parts = line.split('|')
        row_method = parts[1].strip()
        
        log_method = None
        for k, v in method_map.items():
            if v == row_method:
                log_method = k
                break
                
        if log_method and current_clients in results and log_method in results[current_clients]:
            for i, col_name in enumerate(columns):
                if col_name in ["Method", "Rank"]: continue
                col_key = col_name 
                if col_key in results[current_clients][log_method]:
                    val, is_fin = results[current_clients][log_method][col_key]
                    if is_fin:
                        parts[i+1] = f" {val:.4f} ".center(len(parts[i+1]))
                    else:
                        parts[i+1] = f" *{val:.4f}* ".center(len(parts[i+1]))
        
        new_lines.append('|'.join(parts))
    else:
        if in_table:
            in_table = False
        new_lines.append(line)

with open("final_performance_table_ema03_with_superglue.md", "w") as f:
    f.writelines(new_lines)

print("Tables updated!")
