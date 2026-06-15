import sys
sys.path = [p for p in sys.path if 'feddora' not in p.lower()]
from datasets import load_dataset

raw = load_dataset("anli")
for split_name in sorted(raw.keys()):
    print(f"{split_name}: {len(raw[split_name])} samples")

train_r1 = len(raw["train_r1"])
train_r2 = len(raw["train_r2"])
train_r3 = len(raw["train_r3"])
total = train_r1 + train_r2 + train_r3
print(f"\nCombined Train: {train_r1}+{train_r2}+{train_r3}={total}")

test_r1 = len(raw["test_r1"])
test_r2 = len(raw["test_r2"])
test_r3 = len(raw["test_r3"])
print(f"Test R1: {test_r1}")
print(f"Test R2: {test_r2}")
print(f"Test R3: {test_r3}")

n_clients = 3
batch_size = 32
local_steps = 50
per_client_avg = total // n_clients
per_round_data = local_steps * batch_size

print(f"\n=== Client Data Allocation (n_clients={n_clients}, beta=0.5, non-IID) ===")
print(f"Total train samples: {total}")
print(f"Average per client (uniform): {per_client_avg}")
print(f"Per-round per client (local_steps={local_steps} x batch_size={batch_size}): {per_round_data}")
print(f"% of client data used per round: {per_round_data / per_client_avg * 100:.1f}%")
