import os
import re
import json

log_dirs = [
    "/home/hbkim/python/research-feddora/feddora_share/logs/roberta_anli_experiments"
]

log_files = []
for d in log_dirs:
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".log"):
                log_files.append(os.path.join(root, f))

alpha = 0.3
results = {}

metrics = [
    ("anli_avg", r'Global Model Test accuracy \(ANLI Average\): ([\d\.]+)'),
    ("anli_r1", r'Global Model Test accuracy \(R1\): ([\d\.]+)'),
    ("anli_r2", r'Global Model Test accuracy \(R2\): ([\d\.]+)'),
    ("anli_r3", r'Global Model Test accuracy \(R3\): ([\d\.]+)')
]

for lf in log_files:
    with open(lf, "r") as f:
        content = f.read()
    
    exp_name = os.path.basename(lf).replace(".log", "")
    
    for metric_name, regex in metrics:
        matches = re.findall(regex, content)
        if not matches:
            continue
            
        accs = [float(m) for m in matches]
        
        if len(accs) >= 20:
            emas = []
            for i, acc in enumerate(accs):
                if i == 0:
                    emas.append(acc)
                else:
                    emas.append(alpha * acc + (1 - alpha) * emas[-1])
            
            if "anli" not in results:
                results["anli"] = {}
            if exp_name not in results["anli"]:
                results["anli"][exp_name] = {}
            
            results["anli"][exp_name][metric_name] = emas[19] # Round 20 is index 19

print(json.dumps(results, indent=4))
