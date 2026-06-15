# RAVAN Table 2 재현 실험 구현 Walkthrough

## 개요

RAVAN 논문(NeurIPS 2025)의 **Table 2: Performance comparison on CIFAR-100 and SVHN**을 재현하기 위한 코드를 구현했습니다.

---

## 변경된 파일

### 신규 파일

| 파일 | 설명 |
|---|---|
| [svhn.py](file:///home/hbkim/python/research-feddora/feddora_share/datasets/svhn.py) | SVHN 데이터셋 래퍼 (`.targets`, `.num_classes` 인터페이스) |
| [run_vit_cifar100_svhn_experiments.py](file:///home/hbkim/python/research-feddora/feddora_share/run_vit_cifar100_svhn_experiments.py) | 288개 실험 자동 실행 스크립트 |

### 수정된 파일

| 파일 | 변경 내용 |
|---|---|
| [peft_utils.py](file:///home/hbkim/python/research-feddora/feddora_share/peft_utils.py) | `RAVANLinear` 클래스 추가, `inject_peft`에 `target_modules`/`ravan` 지원, `ravan_aggregate()` 함수 |
| [main.py](file:///home/hbkim/python/research-feddora/feddora_share/main.py) | `--peft ravan`, `--ravan_heads`, `--ravan_init`, `--target_modules` 인자 추가, RAVAN 집계 로직 |
| [utils.py](file:///home/hbkim/python/research-feddora/feddora_share/utils.py) | SVHN 데이터셋 로딩, `vit-base` 모델 초기화 지원 |

---

## 실험 하이퍼파라미터 요약

### 공통 설정

| 항목 | 값 |
|---|---|
| **모델** | ViT-B/16 (`vit_base_patch16_224`, 85M params) |
| **데이터셋** | CIFAR-100 (100 classes), SVHN (10 classes) |
| **클라이언트 수** | 20 또는 50 |
| **라운드당 참여** | 3 clients (fraction=0.15 for 20c, 0.06 for 50c) |
| **Non-IID** | Dirichlet α=0.3 |
| **배치 사이즈** | 32 |
| **로컬 steps** | 50 mini-batch iterations |
| **통신 라운드** | 50 |
| **옵티마이저** | AdamW (momentum=0.9) |
| **스케줄러** | Cosine |
| **LoRA 타겟** | `query`, `value` (ViT self-attention만) |
| **Seeds** | 42, 43, 44 (3회 평균) |

### Method별 설정

| Method | PEFT Type | Aggregation | 특이사항 |
|---|---|---|---|
| **FedIT** | LoRA (A+B trainable) | FedAvg | 표준 LoRA + FedAvg |
| **FedEx-LoRA** | LoRA (A+B trainable) | Exact Aggregation | W0에 잔차 영구 누적 |
| **FFA-LoRA** | LoRA (B only trainable) | FedAvg(B) | A frozen, rank 2× |
| **FlexLoRA** | LoRA (A+B trainable) | SVD re-decomposition | W_k 재구성 → FedAvg → SVD |
| **FeDoRA** | DoRA (A+B+m trainable) | SVD-A + Least Squares | A: SVD 추출, B: LS 계산 |
| **RAVAN** | Multi-head (H_i + s_i) | Weighted avg of s_i*H_i | B_i, A_i frozen (Gram-Schmidt) |

### Parameter Budget별 Rank 설정

| Budget | FedIT/FedEx/FlexLoRA/FeDoRA | FFA-LoRA | RAVAN |
|---|---|---|---|
| Lower (~1.2M) | r=32 | r=64 | 4 heads × r=110 |
| Higher (~2.4M) | r=64 | r=128 | 4 heads × r=156 |

### Method별 최적 Learning Rate

**CIFAR-100:** FedIT=5e-3, FedEx-LoRA=1e-3, FFA-LoRA=1e-2, FlexLoRA/FeDoRA=5e-3, RAVAN=5e-4

**SVHN:** FedIT=1e-3, FedEx-LoRA=1e-3, FFA-LoRA=5e-3~1e-3, FlexLoRA/FeDoRA=1e-3, RAVAN=5e-4

---

## RAVAN 구현 상세

### RAVANLinear 아키텍처

```
Forward:  y = W0·x + Σ_{i=1}^{H} s_i · B_i · H_i · A_i · x · (α/r)
```

- **B_i** (out×r): Gram-Schmidt 직교 초기화, frozen
- **A_i** (r×in): Gram-Schmidt 직교 초기화, frozen
- **H_i** (r×r): zero 초기화, **trainable** (core parameter)
- **s_i** (scalar): 1로 초기화, **trainable** (scaling factor)

### 집계 (Algorithm 1)

1. 각 클라이언트에서 `s_i * H_i` 수집
2. 가중 평균: `H_new_i = Σ freq_c * (s_{c,i} * H_{c,i})`
3. 새 글로벌 H_i = averaged product, s_i = 1로 reset

### Trainable 파라미터 수 (per layer)

```
Per layer: H heads × (r² + 1) params
예: 4 × (110² + 1) = 48,404 params/layer
총: 48,404 × 24 layers × 2 (q,v) = 2,323,392 params (for lower budget)
```

---

## 실행 방법

```bash
# 전체 실험 실행 (288개 실험, GPU 0,1 사용)
source activate && python3 run_vit_cifar100_svhn_experiments.py

# 단일 실험 예시 (RAVAN, CIFAR-100, non-IID, 20 clients)
source activate && python3 main.py \
  --model vit-base --dataset cifar100 --datadir /mnt/data1 \
  --round 50 --epochs 1 --local_steps 50 \
  --n_clients 20 --sample_fraction 0.15 \
  --partition noniid --beta 0.3 \
  --peft ravan --lora_r 110 --lora_alpha 110 --ravan_heads 4 \
  --target_modules query value --ft_classifier \
  --optimizer adamw --lr 5e-4 --scheduler cosine \
  --batch_size 32 --seed 42 \
  --logdir ./logs/vit_cifar100_svhn_experiments/cifar100_lower
```

---

## Dry-run 검증 결과

```
✅ 모델: ViT-B/16 (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
✅ 데이터: CIFAR-100, 5 clients, non-IID (β=0.3)
✅ PEFT: RAVAN (4 heads, r=110)
✅ 학습: 3 clients × 3 steps → avg_loss 5.44→5.06→4.84 (정상 감소)
✅ 집계: ravan_aggregate 정상 실행 (s_i*H_i 평균, s_i reset)
✅ 평가: Global Test acc=0.0103 (1 round, 3 steps이므로 정상)
✅ 총 소요: ~229초
```
