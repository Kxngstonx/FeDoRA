import re
from datetime import datetime

log_file = "logs/vit_cifar100_svhn_experiments/cifar100_lower/cifar100_20c_iid_lower_FlexLoRA_seed42.log"

times = []
with open(log_file, 'r') as f:
    for line in f:
        # Example: [06-15 05:03:59] ROUND 1/50
        match = re.search(r'\[06-15 (\d{2}:\d{2}:\d{2})\].*ROUND (\d+)/50', line)
        if match:
            time_str = match.group(1)
            t = datetime.strptime(time_str, "%H:%M:%S")
            times.append(t)

if len(times) >= 2:
    diffs = [(times[i] - times[i-1]).total_seconds() for i in range(1, len(times))]
    avg_sec_per_round = sum(diffs) / len(diffs)
    
    total_sec_per_exp = avg_sec_per_round * 50
    # 48 total experiments, 2 running at a time -> 24 batches
    total_sec_all = total_sec_per_exp * 24
    
    # Time passed for the first batch
    passed_sec = (times[-1] - times[0]).total_seconds()
    # If 2 finished, and 2 running, we have 44 left (22 batches)
    # The current running ones have some time left.
    current_round = len(times)
    sec_left_current = (50 - current_round) * avg_sec_per_round
    
    total_remaining_sec = sec_left_current + (22 * total_sec_per_exp)
    
    hours = total_remaining_sec / 3600
    
    print(f"Avg sec per round: {avg_sec_per_round:.1f}s")
    print(f"Time per experiment (50 rounds): {total_sec_per_exp/60:.1f} minutes")
    print(f"Total remaining experiments: 46 (2 running, 44 waiting)")
    print(f"Estimated remaining time: {hours:.2f} hours")
else:
    print("Not enough rounds to calculate time.")
