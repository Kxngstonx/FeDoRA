# Implementation Plan: Rollback FedALS & Enforce Fair FedEx-LoRA

## 1. Rollback FedALS (Restore FeDoRA's original SVD)
**Rationale:** As pointed out, replacing SVD with ALS fundamentally changes the proposed FeDoRA method. SVD is the core component of FeDoRA's aggregation.
**Action:**
- Revert `peft_utils.py` to remove the `als_a` aggregation logic.
- Revert `main.py` to remove the `--flex_lora_als` argument.
- Revert `run_vit_cifar100_svhn_experiments.py` to restore `--flex_lora_svd_a` for FeDoRA.

## 2. Enforce PEFT Communication Constraints on FedEx-LoRA
**Rationale:** Currently, FedEx-LoRA permanently updates the frozen `W0` on the server and then *broadcasts the entire updated $W0$ back to the clients* every round. This uses $O(|W0|)$ communication bandwidth, which is effectively full-rank fine-tuning and violates the PEFT constraints.
**Action:**
- In `main.py`, remove the logic that updates `global_w` with the modified `W0` (around line 424).
- By doing this, the server will still maintain its updated `W0` for evaluation, but the **clients will only receive $B_{avg}$ and $A_{avg}$**, strictly enforcing the $O(r)$ communication limit.
- This will place FedEx-LoRA on a completely fair and equal playing field with FeDoRA.

## Proposed Code Changes
### [MODIFY] [main.py](file:///home/hbkim/python/research-feddora/feddora_share/main.py)
- Remove `--flex_lora_als` flag.
- Remove `als_a` argument passing to `flex_lora_aggregate()`.
- Delete the `for k, v in global_model.state_dict().items(): global_w[k] = v.clone()` block in the FedEx-LoRA section.

### [MODIFY] [peft_utils.py](file:///home/hbkim/python/research-feddora/feddora_share/peft_utils.py)
- Remove the `als_a` logic block from `flex_lora_aggregate()`.

### [MODIFY] [run_vit_cifar100_svhn_experiments.py](file:///home/hbkim/python/research-feddora/feddora_share/run_vit_cifar100_svhn_experiments.py)
- Revert FeDoRA's argument back to `--peft dora --flex_lora --flex_lora_svd_a ...`

## User Review Required
> [!IMPORTANT]
> 연구원님 말씀이 맞습니다. FeDoRA의 핵심 아이디어인 SVD를 제외하는 것은 본래 기법을 훼손하는 것이므로 전면 롤백하겠습니다. 대신, FedEx-LoRA가 클라이언트에게 $W0$를 매번 다운로드 시키는 반칙 코드를 제거하여, 완전히 동등한 PEFT 통신량 $O(r)$ 환경에서 FeDoRA와 진검승부를 할 수 있도록 수정하겠습니다. 이 계획을 승인해 주시면 즉시 롤백 및 FedEx-LoRA 페널티 조정을 진행하겠습니다.
