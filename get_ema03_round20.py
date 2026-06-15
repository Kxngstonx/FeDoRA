import os
import re

log_dirs = [
    "/home/hbkim/python/research-feddora/feddora_share/logs/roberta_lora_experiments",
    "/home/hbkim/python/research-feddora/feddora_share/logs/roberta_glue_experiments",
    "/home/hbkim/python/research-feddora/feddora_share/logs/roberta_superglue_experiments",
    "/home/hbkim/python/research-feddora/feddora_share/logs/roberta_anli_experiments",
    "/home/hbkim/python/research-feddora/feddora_share/logs/roberta_mmlu_experiments"
]

log_files = []
for d in log_dirs:
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".log"):
                log_files.append(os.path.join(root, f))

alpha = 0.3
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
    
    if len(accs) >= 20:
        emas = []
        for i, acc in enumerate(accs):
            if i == 0:
                emas.append(acc)
            else:
                emas.append(alpha * acc + (1 - alpha) * emas[-1])
        
        if dataset not in results:
            results[dataset] = {}
        
        results[dataset][exp_name] = emas[19] # Round 20 is index 19

import json
print(json.dumps(results, indent=4))
