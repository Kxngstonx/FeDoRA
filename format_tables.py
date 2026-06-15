import json

results = {
    'FeDoRA': {'cifar100_20c_noniid': (0.8113, True), 'cifar100_50c_noniid': (0.8098, True), 'cifar100_20c_iid': (0.832, True), 'cifar100_50c_iid': (0.8278, True), 'svhn_20c_noniid': (0.505, True), 'svhn_20c_iid': (0.552, True)},
    'RAVAN': {'cifar100_50c_noniid': (0.6811, True), 'cifar100_20c_noniid': (0.7314, True), 'cifar100_20c_iid': (0.7826, True), 'cifar100_50c_iid': (0.7526, True), 'svhn_20c_iid': (0.4386, True), 'svhn_20c_noniid': (0.3425, True)},
    'FFA-LoRA': {'cifar100_50c_noniid': (0.8056, True), 'cifar100_50c_iid': (0.8244, True), 'cifar100_20c_noniid': (0.8141, True), 'cifar100_20c_iid': (0.8311, True), 'svhn_50c_iid': (0.4898, False), 'svhn_20c_iid': (0.552, True), 'svhn_20c_noniid': (0.505, True)},
    'FedEx-LoRA': {'cifar100_50c_iid': (0.8244, True), 'cifar100_20c_noniid': (0.8141, True), 'cifar100_20c_iid': (0.8311, True), 'cifar100_50c_noniid': (0.8056, True), 'svhn_20c_noniid': (0.505, True), 'svhn_50c_iid': (0.5464, True), 'svhn_20c_iid': (0.552, True)},
    'FlexLoRA': {'cifar100_20c_noniid': (0.8113, True), 'cifar100_50c_iid': (0.8278, True), 'cifar100_20c_iid': (0.832, True), 'cifar100_50c_noniid': (0.8098, True), 'svhn_20c_iid': (0.552, True), 'svhn_20c_noniid': (0.505, True)},
    'FedIT': {'cifar100_50c_iid': (0.8278, True), 'cifar100_20c_iid': (0.832, True), 'cifar100_50c_noniid': (0.8098, True), 'cifar100_20c_noniid': (0.8113, True), 'svhn_20c_noniid': (0.505, True), 'svhn_50c_iid': (0.5464, True), 'svhn_20c_iid': (0.552, True)}
}

methods_order = ['FeDoRA', 'FedEx-LoRA', 'FedIT', 'FlexLoRA', 'FFA-LoRA', 'RAVAN']
method_names = {
    'FeDoRA': '**🌟 Ours (FeDoRA)**',
    'FedEx-LoRA': 'FedEx-LoRA',
    'FedIT': 'FedIT',
    'FlexLoRA': 'FlexLoRA',
    'FFA-LoRA': 'FFA-LoRA',
    'RAVAN': 'RAVAN'
}

cols = [
    ('cifar100', 'iid'),
    ('cifar100', 'noniid'),
    ('svhn', 'iid'),
    ('svhn', 'noniid')
]

for clients in ['20c', '50c']:
    print(f"### {clients.replace('c', '')} Clients Setting")
    print(f"| Method                    | CIFAR-100 (IID) | CIFAR-100 (Non-IID) | SVHN (IID) | SVHN (Non-IID) |")
    print(f"|---------------------------|-----------------|---------------------|------------|----------------|")
    for method in methods_order:
        row = []
        for ds, part in cols:
            key = f"{ds}_{clients}_{part}"
            if key in results[method]:
                acc, finished = results[method][key]
                acc_str = f"{acc:.4f}"
                if not finished:
                    acc_str = f"*{acc_str}*"
                row.append(acc_str)
            else:
                row.append("-")
        print(f"| {method_names[method]:<25} | {row[0]:>15} | {row[1]:>19} | {row[2]:>10} | {row[3]:>14} |")
    print()
