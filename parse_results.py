import json
import glob
import os

results = {}
for filepath in glob.glob("./logs/vit_cifar100_svhn_experiments/**/*.json", recursive=True):
    filename = os.path.basename(filepath)
    # filename format: dataset_50c_partition_lower_Method_seed42.json
    parts = filename.replace('.json', '').split('_')
    if len(parts) >= 6:
        dataset = parts[0]
        partition = parts[2]
        method = parts[4]
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            max_acc = max([round_data.get('test_acc', 0.0) for round_data in data])
            
            if method not in results:
                results[method] = {}
            key = f"{dataset} ({partition.upper()})"
            results[method][key] = max_acc
        except Exception as e:
            pass

print(json.dumps(results, indent=4))
