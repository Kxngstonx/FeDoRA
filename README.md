# FedDoRA: Federated Learning with Decoupled Weight-Decomposed Low-Rank Adaptation

This repository contains the essential files to run the FedDoRA experiments, which focus on Parameter-Efficient Fine-Tuning (PEFT) methods—specifically LoRA and DoRA (Weight-Decomposed Low-Rank Adaptation)—in a Federated Learning environment under Non-IID data conditions.

## Project Overview

In conventional Federated Learning (FL), transmitting full model weights (Full FT) between clients and the server poses a massive communication bottleneck. While methods like LoRA reduce this overhead by training low-rank matrices (A and B), DoRA extends this by separating the magnitude ($m$) and direction ($V$) of the weights.

This research investigates the dynamics of **DoRA** in a federated setting, proposing novel methods such as **Decoupled DoRA**, **Cosine Re-calibration**, and **Magnitude Warm-up** to mitigate client drift caused by heterogeneous (Non-IID) data distributions (simulated via Dirichlet distributions with varying $\beta$ parameters).

Crucially, in our experiments, the $A$ matrix in LoRA/DoRA is initialized with a Gaussian distribution and **frozen (requires_grad=False)**, ensuring that the alignment of dimensions is perfectly preserved during the server's FedAvg aggregation phase. Only the $B$ matrix and the magnitude vector $m$ are trained.

---

## Key Files & Directory Structure

* **`main.py`**
  The primary entry point for the Federated Learning simulation. It initializes the global model, distributes it to clients, triggers local training (`train.py`), aggregates the updated weights (FedAvg), and evaluates the model. It also contains the implementations for our proposed aggregation heuristics (Cosine Re-calibration).

* **`train.py`**
  Handles the local client training loop. It processes the dataset, applies the chosen optimizer, and computes the loss. It includes logic for **Magnitude Warm-up**, where only the $m$ parameter is updated for an initial fraction of the training steps before unfreezing the $B$ matrix.

* **`peft_utils.py`**
  Contains the PyTorch `nn.Module` implementations for `LoRALinear`, `DoRALinear`, `LoRAConv2d`, and `DoRAConv2d`. 
  - *Key Feature:* The $A$ matrices are initialized with a standard normal distribution and explicitly frozen (`requires_grad=False`).

* **`utils.py`**
  Provides utility functions for the experiment pipeline, such as dataset partitioning (Dirichlet Non-IID allocation), logging, metric calculations, and the `DoRASimilarityCalculator` used to track the divergence (L2/KL) of magnitude and direction vectors between clients and the server.

* **`datasets/`**
  Contains scripts to download and preprocess various datasets (e.g., AG News, CIFAR-10, CIFAR-100, Banking77) and split them into localized client shards.

* **`models/`**
  Contains the architectures used for the experiments. Includes integration for Hugging Face transformer models (like `qwen_llm.py`, `roberta-base`), and standard vision models (`resnet_cifar.py`, `mobilenet.py`).

* **`run_fixed_A_experiments.py` & `run_fixed_a_experiments.sh`**
  The automated execution pipeline. `run_fixed_A_experiments.py` uses a multi-threading queue to dispatch tasks across multiple GPUs in parallel. The shell script acts as a simple launcher.

---

## How to Run the Experiments

To replicate the core experiments comparing Full FT, LoRA, Coupled DoRA, Decoupled DoRA, and our proposed methods (Cosine/Warmup) under different Non-IID severities ($\beta = 0.5, 0.3, 0.1$), run the provided shell script:

```bash
bash run_fixed_a_experiments.sh
```

### What this script does:
1. It reads previously established optimal hyper-parameters (from the `logs/decoupled_experiments/` directory if they exist, or uses defaults).
2. It launches 15+ individual Python processes using `main.py`.
3. It efficiently manages your hardware by queuing the tasks sequentially onto available GPUs (defaults to GPU 0 and 1).
4. The output logs for each task will be independently saved in the `logs/fixed_a_experiments/beta_{beta}/` directory.

### Example Direct Execution
If you prefer to run a single experiment manually, you can execute `main.py` directly. For example, to run Decoupled DoRA with Cosine Re-calibration on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --model qwen \
    --dataset ag_news \
    --round 50 \
    --epochs 1 \
    --n_clients 100 \
    --sample_fraction 0.05 \
    --beta 0.3 \
    --partition noniid \
    --peft dora \
    --decoupled_dora True \
    --use_cosine_recal \
    --dora_cos_tau 1.0 \
    --dora_cos_gamma 0.0
```
