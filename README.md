<div align="center">

# FeDoRA: Federated Weight-Decomposed Low-Rank Adaptation

**Federated fine-tuning of foundation models under data & system heterogeneity.**
Comparison of parameter-efficient fine-tuning (PEFT) methods on **NLP** (RoBERTa) and **Vision** (ViT) tasks.

</div>

---

## 1. Overview

This repository implements and compares federated PEFT methods. The proposed method is **FeDoRA**.

| Method | Idea | Key flag |
|--------|------|----------|
| **FeDoRA** (ours) | Client = **DoRA** (weight = magnitude × direction); Server = full-weight FedAvg + **SVD-A** aggregation (common direction `A` via SVD of stacked client `A`, then least-squares `B`) | `--peft dora --flex_lora --flex_lora_svd_a` |
| FlexLoRA | LoRA + reconstruct ΔW → FedAvg → SVD re-decompose | `--peft lora --flex_lora` |
| FedEx-LoRA | Exact aggregation: add residual `mean(BᵢAᵢ) − B̄Ā` to W₀ | `--peft lora --fedex_lora --trainable_A` |
| FFA-LoRA | Freeze `A`, train only `B` | `--peft lora --flex_lora --flex_lora_freeze_a` |
| FedIT | Vanilla LoRA + FedAvg of `A`, `B` | `--peft lora --trainable_A` |
| RAVAN (NeurIPS'25) | Multi-head augmented LoRA (`B H A`), train core `H` | `--peft ravan` |

FeDoRA keeps the base weights **frozen** and communicates only **O(r)** low-rank factors, unlike FedEx-LoRA which mutates W₀ (see [§6](#6-fedex-lora-fair-communication-variants)).

---

## 2. Repository Structure

| File | Role |
|------|------|
| **[`main.py`](./main.py)** | Federated server loop: client selection, weight distribution, aggregation, evaluation, argument parsing. |
| **[`peft_utils.py`](./peft_utils.py)** | PEFT layers (`LoRALinear`, `DoRALinear`, `RAVANLinear`) and server aggregation (`flex_lora_aggregate`, `fedex_lora_aggregate`, `ravan_aggregate`). The most important file for the algorithms. |
| **[`train.py`](./train.py)** | Client-side local training (`train_local_net`), incl. DoRA differential LR/WD (`--dora_m_lr`). |
| **[`datasets/`](./datasets/)** | Dataset loaders (GLUE, SuperGLUE, ANLI, MMLU, CIFAR-100, SVHN, …). |
| `run_roberta_*.py` | **NLP** experiment launchers (see [§4](#4-nlp-tasks-roberta-large)). |
| `run_vit_*.py` | **Vision** experiment launchers (see [§5](#5-vision-tasks-vit-b16)). |
| `run_fedora_gridsearch*.py`, `run_phase_*.py` | **FeDoRA hyperparameter grid search** (see [§7](#7-hyperparameter-grid-search-fedora)). |

Each launcher spreads jobs across GPUs via a thread queue (with optional VRAM gating) and writes per-run logs containing `>> Global Model Test accuracy: …` once per round.

---

## 3. Setup

```bash
source bin/activate          # project venv (PyTorch 2.4 + CUDA 12.1, timm, transformers)
pip install -r requirements.txt
```

Shared federated settings (all tasks): `--scheduler cosine`, `--local_steps 50`, 1 local epoch/round,
`--partition noniid --beta <β>` (Dirichlet) or `--partition iid`, and `--ft_classifier`
(the classifier head is fully fine-tuned, not adapted). The pre-trained backbone is fully frozen
(`requires_grad = False`); only the PEFT adapters + classifier head are trained.

---

## 4. NLP Tasks (RoBERTa-large)

**Model:** `roberta-large` · **Clients:** 3 · **Optimizer:** AdamW · **LR:** 1e-4 ·
**Rounds:** 30 · **Rank:** `lora_r = lora_alpha = 8` · **Partition:** Non-IID (Dirichlet β=0.5).

| Benchmark | Datasets | Launcher | Seeds |
|-----------|----------|----------|-------|
| **GLUE** | SST-2, MNLI, QQP | [`run_roberta_glue_experiments.py`](./run_roberta_glue_experiments.py) | 42 |
| **SuperGLUE** | BoolQ, MultiRC, ReCoRD, WiC | [`run_roberta_superglue_experiments.py`](./run_roberta_superglue_experiments.py) | 42, 43, 44 |
| **ANLI** | R1, R2, R3 (adversarial NLI) | [`run_roberta_anli_experiments.py`](./run_roberta_anli_experiments.py) | 42 |
| **MMLU** | MMLU | [`run_roberta_mmlu_experiments.py`](./run_roberta_mmlu_experiments.py) | 42 |

```bash
python3 run_roberta_glue_experiments.py        # GLUE: SST-2 / MNLI / QQP × all methods
python3 run_roberta_superglue_experiments.py   # SuperGLUE multi-seed (42/43/44)
python3 run_roberta_anli_experiments.py
python3 run_roberta_mmlu_experiments.py
```

Reported metric: **EMA(0.3) @ Round 20** of global test accuracy (multi-seed averaged for SuperGLUE).

---

## 5. Vision Tasks (ViT-B/16)

**Model:** `vit-base` (timm, ImageNet-21k→1k) · **Clients:** {20, 50} (3 sampled/round) ·
**Optimizer:** Adam (`--reg 0`) · **Rounds:** 50 (main) · **Batch:** 32 · **Target:** fused `qkv` ·
**Partition:** IID and Non-IID (Dirichlet β=0.3). Launcher: [`run_vit_cifar100_svhn_experiments.py`](./run_vit_cifar100_svhn_experiments.py)
(48 combinations: Dataset × Client count × Partition × Method).

| Benchmark | Classes | Per-method rank (lower-budget ≈1.2M params) |
|-----------|---------|----------------------------------------------|
| **CIFAR-100** | 100 | FeDoRA/FedIT/FlexLoRA `r=32`; FFA-LoRA `r=64`; RAVAN `r=110, 4 heads` |
| **SVHN** | 10 | (same as above) |

```bash
python3 run_vit_cifar100_svhn_experiments.py   # CIFAR-100 & SVHN × {20,50} clients × {IID,NonIID} × all methods
```

Reported metric: **EMA(0.3) @ Round 50** and/or **Max** test accuracy.
> - `timm`'s ViT fuses Q/K/V into one `qkv` linear → use `--target_modules qkv` (not `query value`).
> - RAVAN's orthogonal initialization fails on CIFAR-100, so it is reported only on SVHN.

---

## 6. FedEx-LoRA Fair-Communication Variants

FedEx-LoRA's "exact aggregation" mutates the frozen base `W₀` and (in the naive implementation)
redistributes the full dense `W₀` every round — **≈19× the downlink** of the other O(r) PEFT methods
for ViT-B `qkv`. Two fairness variants are provided so all methods share an equal communication budget:

| Flag | Behaviour | Per-round downlink |
|------|-----------|--------------------|
| `--fedex_freeze_w0` | Keep `W₀` frozen (drop residual) → reduces to FedIT | 1× |
| `--fedex_lowrank_comm --fedex_res_rank R` | Carry the accumulated residual as a **rank-R low-rank adapter** (`res_A`,`res_B`) instead of dense `W₀`; set `lora_r + R` = baseline rank for an equal budget | ≈1–2× |

Launchers: [`run_vit_fedex_fair_experiments.py`](./run_vit_fedex_fair_experiments.py) (frozen-W₀),
[`run_vit_fedex_faircomm_experiments.py`](./run_vit_fedex_faircomm_experiments.py) (low-rank residual),
[`run_vit_fedex_scenarios_experiments.py`](./run_vit_fedex_scenarios_experiments.py) (all scenarios).

---

## 7. Hyperparameter Grid Search (FeDoRA)

FeDoRA is tuned **without changing the algorithm** (`--peft dora --flex_lora --flex_lora_svd_a`,
no warm-up). Only hyperparameters are swept. Candidate values follow common ranges from
LoRA (Hu et al., 2021), DoRA (Liu et al., 2024), FedIT, and RAVAN.

### Search space (per dataset)

| Hyperparameter | Flag | Candidates | # |
|----------------|------|-----------|---|
| Learning rate | `--lr` | `5e-4, 1e-3, 5e-3, 1e-2` | 4 |
| LoRA scaling | `--lora_alpha` (r=32 → α/r = 0.5 / 1 / 2) | `16, 32, 64` | 3 |
| DoRA magnitude LR | `--dora_m_lr` (separate LR for magnitude `m`; `same` = share `--lr`) | `same, 1e-3, 1e-4` | 3 |

→ **4 × 3 × 3 = 36 combinations / partition**, run for **{IID, Non-IID} = 72 runs / dataset**, each **30 rounds**.
Selection metric: **EMA(0.3) @ Round 30**.
*(Rule of thumb: 3–5 candidates per hyperparameter over 2–3 key hyperparameters → ~9–36 combinations is a typical "medium" grid.)*

> `--dora_m_lr` is **not** warm-up. It places the DoRA magnitude vector `m` (from `W = m · V/‖V‖`)
> into a separate optimizer group with its own learning rate throughout training ([`train.py`](./train.py)),
> letting the magnitude and direction adapt at different rates.

### Launchers

```bash
# General, dataset-agnostic, with quick / medium / full presets
python3 run_fedora_gridsearch.py --preset medium --dry-run        # show plan & combo count
python3 run_fedora_gridsearch.py --preset medium --datasets cifar100 --partitions noniid
python3 run_fedora_gridsearch.py --summarize                      # rank the best config per environment

# CIFAR-100, 20 clients, 30 rounds (72 runs)
python3 run_fedora_gridsearch_cifar30.py
python3 run_fedora_gridsearch_cifar30.py --summarize

# SVHN 20c grid → then 50-client runs with the best HP (orchestrated, background-friendly)
python3 run_phase_svhn20_then_50c.py
```

Logs go to `logs/fedora_gridsearch*/`; completed runs (≥ target rounds) are auto-skipped, so the
search is **resumable** after interruption.

### Findings (CIFAR-100, 20 clients, EMA@30)

- **Learning rate dominates**: `1e-3 (~0.92) > 5e-4 (~0.91) ≫ 5e-3 (unstable) ≫ 1e-2 (diverges, 0.1–0.3)`.
  The original FeDoRA CIFAR-100 setting (`5e-3`) was simply too large — proper tuning lifts FeDoRA from
  ~0.86 to **0.92**, on par with / above the strongest baselines.
- **Balanced best**: `--lr 1e-3 --lora_alpha 32 --dora_m_lr 1e-3` (highest IID+Non-IID average).
- **`dora_m_lr`** has a small effect in the stable regime (±0.002), mattering only when the LR is already too large.

---

## 8. Parsing Results

Each `*.log` contains `>> Global Model Test accuracy: <acc>` once per round.
The reported number is the **EMA** `eₜ = 0.7·eₜ₋₁ + 0.3·xₜ` at the target round, and/or the per-run **Max**.

```bash
python3 parse_results_new.py     # recursively parse logs/ and extract final accuracies
```

Full result tables (GLUE / SuperGLUE / ANLI / MMLU / Vision + grid search) are maintained in
[`final_performance_table_ema03_with_superglue.md`](./final_performance_table_ema03_with_superglue.md).

---

## Implementation Notes

- **Optimizer:** Vision uses standard `Adam` with `weight_decay=0` (`--reg 0`); NLP uses `AdamW`, matching the reference papers.
- **ViT `qkv` fusion:** parameter budgets are verified to match the paper's lower-budget (~1.2M) setting with `--target_modules qkv`.
- **Base freezing:** only PEFT adapters + the final classifier head are trainable; the backbone is frozen.
