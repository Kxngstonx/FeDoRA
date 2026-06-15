# FeDoRA & RAVAN Federated Learning Experiments

This repository contains the codebase for federated fine-tuning experiments, comparing various Parameter-Efficient Fine-Tuning (PEFT) methods under data and computational heterogeneity.

The implemented PEFT methods include:
- **FeDoRA** (Ours)
- **FlexLoRA**
- **FedEx-LoRA**
- **FFA-LoRA**
- **FedIT**
- **RAVAN** (NeurIPS 2025)

## 🚀 How to Run the Experiments

To start the Vision benchmarks (CIFAR-100 & SVHN) with ViT-B/16 using the identical settings from the RAVAN paper (Lower Budget setting):

```bash
# Activate your virtual environment (if any)
source activate

# Run the experiment orchestrator
python3 run_vit_cifar100_svhn_experiments.py
```

*Note: This script will automatically schedule and run 48 experimental combinations (Dataset × Client Counts × Partition Type × PEFT Methods) across available GPUs by managing VRAM.*

## 📂 Code Structure: Where to Look?

If you want to understand or modify the code, here are the core files to investigate:

### 1. ⚙️ Experiment Orchestration
- **[`run_vit_cifar100_svhn_experiments.py`](./run_vit_cifar100_svhn_experiments.py)**  
  The main entry point for Vision experiments. It defines the hyperparameters (ranks, learning rates for each PEFT method) corresponding to the "Lower Budget" (~1.2M parameters) setting in the paper. It builds the bash commands and pushes them to a thread queue for parallel execution on GPUs.

### 2. 🧠 PEFT Implementations & Aggregation
- **[`peft_utils.py`](./peft_utils.py)**  
  This is the most critical file for understanding the PEFT methods. It includes:
  - `inject_peft()`: Logic to inject LoRA, DoRA, or RAVAN into the base models. Note that for `timm` ViT models, the target module is `qkv`.
  - `RAVANLinear`, `LoRALinear`, `DoRALinear`: PyTorch `nn.Module` classes defining the forward passes of the adapters.
  - Server Aggregation Functions: `fedex_lora_aggregate()`, `flex_lora_aggregate()`, and `ravan_aggregate()`.

### 3. 🌐 Federated Learning Loop
- **[`main.py`](./main.py)**  
  The core federated learning server script. It handles:
  - Global model initialization and base weight freezing.
  - Client selection and distributing weights to local clients.
  - Calling the corresponding aggregation function from `peft_utils.py` once clients return their trained weights.

### 4. 💻 Client Local Training
- **[`train.py`](./train.py)**  
  Contains `train_local_net()`, which executes the local training iterations for each selected client using cross-entropy loss and the Adam optimizer (with `weight_decay=0`).

### 5. 📊 Result Parsing & Logging
- **[`parse_results_new.py`](./parse_results_new.py)**  
  After experiments finish, run this script to recursively parse `.log` files in the `logs/` directory and extract the final "Global Model Test accuracy".
- **[`final_performance_table_ema03_with_superglue.md`](./final_performance_table_ema03_with_superglue.md)**  
  A markdown file where the final summarized performance tables are maintained.

## ⚠️ Important Implementation Notes

- **Optimizer:** We use standard `Adam` with `weight_decay=0` (`--reg 0`) to match the exact setup of the reference papers, rather than `AdamW`.
- **ViT qkv Fusion:** `timm`'s `vit_base_patch16_224` fuses Query, Key, and Value projections into a single `qkv` linear layer. Consequently, the `target_modules` for PEFT injection is set to `qkv` (as opposed to separate `query` and `value` modules). The parameter budgets have been verified to match the paper's expectations.
- **Base Model Freezing:** The pre-trained ViT backbone is completely frozen (`requires_grad = False`), while only the PEFT adapters and the final classifier head are trained.
