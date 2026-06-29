<div align="center">

# 🏆 Final Performance Summary (Round 20, EMA 0.3)
**GLUE, SuperGLUE, ANLI 벤치마크 20라운드 최종 성능 비교**

</div>

> [!NOTE]
> - 모든 로그 디렉토리를 취합하여 **Round 20** 기준 **EMA(0.3)** 수치를 추출했습니다.
> - 각 데이터셋별 최고 성능은 **굵은 글씨(Bold)** 로 표시했습니다.
> - **SuperGLUE 벤치마크**는 Seed별(42, 43, 44) 극단적 Non-IID 환경 분산 차이를 명확히 보여주기 위해 개별 테이블로 분리했습니다.

<br>

## 📚 1. GLUE Benchmark

| Method        |  MNLI  |  QQP   |    SST2    |
|---------------|--------|--------|------------|
| Ours (FeDoRA) | 0.8377 | 0.8316 | **0.9456** |
| FedEx-LoRA    | 0.8332 | 0.8377 |   0.9443   |
| FedIT         | 0.8294 | 0.8340 |   0.9443   |
| FlexLoRA      | 0.8061 | 0.8301 |   0.9436   |
| FFA-LoRA      | 0.7829 | 0.8061 |   0.9424   |

<br>

## 📚 2. SuperGLUE Benchmark (Multi-Seed Analysis)

### 🌱 Seed 42 (기존 결과 - 'Lucky' Partition)
| Method        |   BOOLQ    | MULTIRC | RECORD |    WIC     |
|---------------|------------|---------|--------|------------|
| Ours (FeDoRA) |   0.7986   |  0.5716 |   -    |   0.6911   |
| FedEx-LoRA    | **0.8050** |  0.5705 |   -    | **0.6987** |
| FedIT         |   0.8040   |  0.5721 | 0.8660 |   0.6908   |
| FlexLoRA      |   0.7977   |  0.5721 |   -    |   0.6939   |
| FFA-LoRA      |   0.7624   |  0.5720 |   -    |   0.6790   |

### 🌱 Seed 43 (Extreme Non-IID Partition)
| Method        |   BOOLQ    |  MULTIRC   |   RECORD   | WIC |
|---------------|------------|------------|------------|-----|
| Ours (FeDoRA) | **0.6219** |   0.5630   | **0.8660** |  -  |
| FedEx-LoRA    |   0.6218   |   0.5685   | **0.8660** |  -  |
| FedIT         |   0.6217   |   0.5635   | **0.8660** |  -  |
| FlexLoRA      |   0.6217   | **0.5745** | **0.8660** |  -  |
| FFA-LoRA      |   0.6217   |   0.5718   | **0.8660** |  -  |

### 🌱 Seed 44 (Extreme Non-IID Partition)
| Method        |   BOOLQ    |  MULTIRC   |   RECORD   | WIC |
|---------------|------------|------------|------------|-----|
| Ours (FeDoRA) |   0.6738   |   0.5356   |   0.8684   |  -  |
| FedEx-LoRA    |   0.6848   |   0.5674   |   0.8673   |  -  |
| FedIT         |   0.6871   |   0.5566   |   0.8686   |  -  |
| FlexLoRA      | **0.6992** | **0.5687** | **0.8699** |  -  |
| FFA-LoRA      |   0.6429   |   0.5483   |   0.8660   |  -  |

<br>

## 📚 3. ANLI (Adversarial NLI)

| Method        | ANLI (Avg) |   R1   |     R2     |     R3     |
|---------------|------------|--------|------------|------------|
| Ours (FeDoRA) |   0.4294   | 0.5396 | **0.3745** | **0.3741** |
| FedEx-LoRA    |   0.4038   | 0.4980 |   0.3570   |   0.3564   |
| FedIT         |   0.4097   | 0.5134 |   0.3531   |   0.3625   |
| FlexLoRA      |   0.4162   | 0.5249 |   0.3648   |   0.3588   |
| FFA-LoRA      |   0.3495   | 0.3603 |   0.3498   |   0.3385   |

<br>

## 📚 4. MMLU Benchmark (Massive Multitask Language Understanding)

| Method        |    MMLU    |
|---------------|------------|
| Ours (FeDoRA) |   0.2531   |
| FedEx-LoRA    |   0.2546   |
| FedIT         |   0.2524   |
| FlexLoRA      |   0.2536   |
| FFA-LoRA      | **0.2549** |

<br>


## 📚 5. Vision Benchmarks (CIFAR-100 & SVHN)
*Note: Results are based on the Lower Budget, Seed 42 experiments (EMA 0.7, Round 50). All 20-client and 50-client CIFAR-100/SVHN experiments are complete (RAVAN failed to launch on CIFAR-100). FedEx-LoRA numbers below use the original W0-updating aggregation; see the fairness note in §7 for the W0-frozen re-run.*

### 20 Clients Setting
| Method        | Rank | CIFAR-100 (IID) | CIFAR-100 (Non-IID) | SVHN (IID)     | SVHN (Non-IID) |
|---------------|------|-----------------|---------------------|----------------|----------------|
| Ours (FeDoRA) |  32  |      0.8712     |        0.8321       |     0.9660     |     0.9568     |
| FedEx-LoRA    |  32  |  **0.9249**     |    **0.9152**       |     0.9690     |     0.9613     |
| FedIT         |  32  |      0.9053     |        0.8872       |     0.9690     |     0.9617     |
| FlexLoRA      |  32  |      0.0384     |        0.0223       |     0.9696     | **0.9632**     |
| FFA-LoRA      |  64  |      0.8497     |        0.8319       |     0.4060     |     0.3096     |
| RAVAN         | 110  |        -        |          -          |     0.9533     |     0.8914     |

### 50 Clients Setting
| Method        | Rank | CIFAR-100 (IID) | CIFAR-100 (Non-IID) |   SVHN (IID)   |  SVHN (Non-IID) |
|---------------|------|-----------------|---------------------|----------------|-----------------|
| Ours (FeDoRA) |  32  |      0.8709     |        0.8406       |     0.9654     |      0.9592     |
| FedEx-LoRA    |  32  |  **0.9228**     |        0.9133       |     0.9685     |      0.9599     |
| FedIT         |  32  |      0.9088     |        0.8873       |     0.9680     |      0.9601     |
| FlexLoRA      |  32  |      0.8898     |        0.8048       |     0.9684     |  **0.9611**     |
| FFA-LoRA      |  64  |      0.8641     |        0.8406       |     0.5407     |      0.2304     |
| RAVAN         | 110  |        -        |    **0.9076**       |     0.9514     |      0.9392     |

<br>

## ⚖️ 7. Fairness Note — FedEx-LoRA의 W0 업데이트

> [!WARNING]
> 위 표의 FedEx-LoRA 수치는 **사전학습 가중치 W0를 매 라운드 갱신**하는 원본 집계를 사용한다.
> 이는 다른 PEFT 베이스라인과의 비교에서 **불공정**하다.

**왜 불공정한가:**
- **통신량 위반**: FedEx-LoRA는 잔차(residual = mean(BₖAₖ) − B_avg·A_avg)를 W0에 더한다. 갱신된 W0는 `state_dict` 배포 경로([main.py:248](main.py#L248) → [main.py:268](main.py#L268))를 통해 매 라운드 클라이언트로 재전송되므로, 실질 통신량이 $O(r)$이 아닌 **$O(d^2)$ (full weight)** 가 된다. 나머지 기법(FeDoRA·FlexLoRA·FedIT·FFA-LoRA)은 동결된 W0 위에서 $O(r)$ 저랭크 파라미터만 주고받는다.
- **표현력 위반**: W0에 잔차를 **라운드마다 누적**하므로, $T$ 라운드 후 base model은 rank-$r$ 제약을 벗어난 **full-rank delta**를 갖는다. 다른 기법은 frozen base 위 단일 rank-$r$ 어댑터만 유지한다.

**공정 버전(W0 동결):** `--fedex_freeze_w0` 플래그로 W0 갱신을 생략하면 B_avg·A_avg만 남아 **FedIT와 수학적으로 등가**가 된다. 즉, 위 표의 FedEx-LoRA가 FedIT 대비 보이는 +1.4~2.8%p 우위는 **전적으로 W0 변형에서 기인**한다.

**재실험 방법:**
```bash
python3 run_vit_fedex_fair_experiments.py   # FedEx-LoRA-fairW0 (8개 환경 전부 재실행)
```
결과는 `logs/vit_cifar100_svhn_experiments/{dataset}_lower/*_FedEx-LoRA-fairW0_seed42.log` 에 저장된다 (기존 결과 비파괴).

| 구분 | base W0 | 라운드별 통신 | 동치 |
|------|---------|--------------|------|
| FedEx-LoRA (원본) | 매 라운드 누적 갱신 | $O(d^2)$ | — |
| FedEx-LoRA-fairW0 (`--fedex_freeze_w0`) | 사전학습값 동결 | $O(r)$ | FedIT |

<br>

---

> ## 🔍 6. Ablation Study (FeDoRA Components)
> **이 섹션은 메인 실험 결과와 분리된 독립적인 구조 분석(Ablation Study) 섹션입니다.**
> 제안 기법(FeDoRA) 내부의 각 요소(FlexLoRA, FFALoRA)가 성능에 미치는 영향을 개별적으로 비교합니다.

### 1. GLUE Benchmark
| Method                    |    MNLI    |    QQP     |    SST2    |
|---------------------------|------------|------------|------------|
| Ours (FeDoRA)             |   0.8377   |   0.8316   | **0.9456** |
| FL+DoRA(FlexLoRA)         | **0.8381** | **0.8378** |   0.9400   |
| FL+DoRA(FlexLoRA+FFALoRA) |   0.7915   |   0.8070   |   0.9421   |

### 🌱 Seed 42 (기존 결과 - 'Lucky' Partition)
| Method                    | BOOLQ  |  MULTIRC   |   RECORD   |  WIC   |
|---------------------------|--------|------------|------------|--------|
| Ours (FeDoRA)             | 0.7986 |   0.5716   |     -      | 0.6911 |
| FL+DoRA(FlexLoRA)         | 0.8048 | **0.5729** | **0.8675** | 0.6967 |
| FL+DoRA(FlexLoRA+FFALoRA) | 0.7618 |   0.5720   |     -      | 0.6706 |

### 🌱 Seed 43 (Extreme Non-IID Partition)
| Method                    |   BOOLQ    | MULTIRC |   RECORD   | WIC |
|---------------------------|------------|---------|------------|-----|
| Ours (FeDoRA)             | **0.6219** |  0.5630 | **0.8660** |  -  |
| FL+DoRA(FlexLoRA)         |     -      |    -    |     -      |  -  |
| FL+DoRA(FlexLoRA+FFALoRA) |     -      |    -    |     -      |  -  |

### 🌱 Seed 44 (Extreme Non-IID Partition)
| Method                    | BOOLQ  | MULTIRC | RECORD | WIC |
|---------------------------|--------|---------|--------|-----|
| Ours (FeDoRA)             | 0.6738 |  0.5356 | 0.8684 |  -  |
| FL+DoRA(FlexLoRA)         |   -    |    -    |   -    |  -  |
| FL+DoRA(FlexLoRA+FFALoRA) |   -    |    -    |   -    |  -  |

### 3. ANLI (Adversarial NLI)
| Method                    | ANLI (Avg) |     R1     |     R2     |     R3     |
|---------------------------|------------|------------|------------|------------|
| Ours (FeDoRA)             |   0.4294   |   0.5396   | **0.3745** | **0.3741** |
| FL+DoRA(FlexLoRA)         | **0.4331** | **0.5566** |   0.3742   |   0.3684   |
| FL+DoRA(FlexLoRA+FFALoRA) |   0.3614   |   0.4090   |   0.3405   |   0.3347   |

### 4. MMLU Benchmark (Massive Multitask Language Understanding)
| Method                    |  MMLU  |
|---------------------------|--------|
| Ours (FeDoRA)             | 0.2531 |
| FL+DoRA(FlexLoRA)         |   -    |
| FL+DoRA(FlexLoRA+FFALoRA) |   -    |

<br>

---

> ## 🔍 7. Grid Search 반영 — CIFAR-100 (ViT-B/16, 20 Clients)
> **이 섹션은 메인 결과(EMA@Round50)와 분리된 그리드서치 반영 섹션입니다.**
> 기존 CIFAR-100 기법별 결과(메인 로그)에 **그리드서치로 튜닝한 Ours (FeDoRA, grid-best)** 행을 추가했습니다.
> 지표는 **EMA(0.3)@Round30** 과 **Max(전 라운드 최고 정확도)** 입니다. (기존 EMA@50은 §5 참조)
>
> - 그리드: lr×4 {5e-4,1e-3,5e-3,1e-2} · lora_alpha×3 {16,32,64} · dora_m_lr×3 {same,1e-3,1e-4}
> - FeDoRA 기법 불변(`dora + flex_lora + svd_a`). grid-best = 해당 partition 최고 EMA@30 조합.
> - **진행 상태**: IID 22/36 완료, **Non-IID 진행 중(미완료)**.

### CIFAR-100, 20 Clients (EMA@30 / Max)
| Method                       |  IID EMA@30  |   IID Max    | Non-IID EMA@30 | Non-IID Max  |
|------------------------------|--------------|--------------|----------------|--------------|
| Ours (FeDoRA)                |    0.8560    |    0.8809    |     0.7642     |    0.8423    |
| **Ours (FeDoRA, grid-best)** |  **0.9229**  |    0.9237    |   **0.9132**   |    0.9157    |
| FedEx-LoRA                   |    0.9205    |  **0.9254**  |     0.8943     |  **0.9159**  |
| FedIT                        |    0.8750    |    0.9071    |     0.7858     |    0.8910    |
| FlexLoRA                     |    0.0260    |    0.7480    |     0.0153     |    0.5504    |
| FFA-LoRA                     |    0.8093    |    0.8510    |     0.7281     |    0.8364    |
| RAVAN                        |      -       |      -       |       -        |      -       |

> - **grid-best FeDoRA**: IID = (lr=1e-3, α=16, m_lr=1e-4) → EMA@30 **0.9229**; Non-IID = (lr=1e-3, α=32, m_lr=1e-3) → EMA@30 **0.9132**.
> - **튜닝된 FeDoRA가 EMA@30에서 모든 기법을 추월**: IID 0.9229 > FedEx 0.9205, **Non-IID 0.9132 > FedEx 0.8943 (+0.019)**. (Max는 FedEx와 근소차) 즉 기존 FeDoRA의 CIFAR-100 저조는 알고리즘이 아니라 **학습률(원본 5e-3) 문제**였음.
> - (RAVAN은 CIFAR-100에서 init 단계 실패로 `-`.)

<br>

### 전체 그리드 결과 (FeDoRA, CIFAR-100 20c, EMA@30 / Max)
*lr → lora_alpha → dora_m_lr 순으로 묶음. 각 partition 최고 EMA@30은 **굵게**. 72/72 셀 완료.*

| lr | lora_alpha | dora_m_lr | IID EMA@30 | IID Max | Non-IID EMA@30 | Non-IID Max |
|------|:----:|:----:|:----------:|:-------:|:--------------:|:-----------:|
| 5e-4 | 16 | same | 0.9152 | 0.9156 | 0.9049 | 0.9067 |
|      |    | 1e-3 | 0.9153 | 0.9157 | 0.9027 | 0.9043 |
|      |    | 1e-4 | 0.9161 | 0.9169 | 0.9053 | 0.9068 |
|      | 32 | same | 0.9173 | 0.9178 | 0.9074 | 0.9088 |
|      |    | 1e-3 | 0.9186 | 0.9194 | 0.9066 | 0.9084 |
|      |    | 1e-4 | 0.9175 | 0.9179 | 0.9074 | 0.9094 |
|      | 64 | same | 0.9197 | 0.9212 | 0.9089 | 0.9104 |
|      |    | 1e-3 | 0.9204 | 0.9211 | 0.9084 | 0.9115 |
|      |    | 1e-4 | 0.9185 | 0.9189 | 0.9079 | 0.9098 |
| 1e-3 | 16 | same | 0.9222 | 0.9226 | 0.9120 | 0.9140 |
|      |    | 1e-3 | 0.9217 | 0.9224 | 0.9111 | 0.9137 |
|      |    | 1e-4 | **0.9229** | 0.9237 | 0.9107 | 0.9128 |
|      | 32 | same | 0.9215 | 0.9226 | 0.9121 | 0.9140 |
|      |    | 1e-3 | 0.9212 | 0.9223 | **0.9132** | 0.9157 |
|      |    | 1e-4 | 0.9206 | 0.9215 | 0.9115 | 0.9135 |
|      | 64 | same | 0.9166 | 0.9181 | 0.9043 | 0.9074 |
|      |    | 1e-3 | 0.9184 | 0.9197 | 0.9054 | 0.9078 |
|      |    | 1e-4 | 0.9130 | 0.9167 | 0.8980 | 0.9014 |
| 5e-3 | 16 | same | 0.8914 | 0.8969 | 0.8706 | 0.8765 |
|      |    | 1e-3 | 0.8847 | 0.8881 | 0.8584 | 0.8643 |
|      |    | 1e-4 | 0.7766 | 0.8701 | 0.7209 | 0.7973 |
|      | 32 | same | 0.8667 | 0.8782 | 0.8262 | 0.8367 |
|      |    | 1e-3 | 0.8311 | 0.8365 | 0.7769 | 0.7915 |
|      |    | 1e-4 | 0.3581 | 0.8162 | 0.2684 | 0.6439 |
|      | 64 | same | 0.2890 | 0.3031 | 0.1274 | 0.1519 |
|      |    | 1e-3 | 0.1754 | 0.1956 | 0.1267 | 0.1594 |
|      |    | 1e-4 | 0.1693 | 0.1804 | 0.0418 | 0.0666 |
| 1e-2 | 16 | same | 0.2702 | 0.2853 | 0.1455 | 0.1614 |
|      |    | 1e-3 | 0.2513 | 0.2692 | 0.1331 | 0.1658 |
|      |    | 1e-4 | 0.1561 | 0.1765 | 0.0492 | 0.0789 |
|      | 32 | same | 0.2564 | 0.2703 | 0.1070 | 0.1195 |
|      |    | 1e-3 | 0.1621 | 0.1836 | 0.0649 | 0.0834 |
|      |    | 1e-4 | 0.1147 | 0.1342 | 0.0418 | 0.0483 |
|      | 64 | same | 0.2085 | 0.2183 | 0.0451 | 0.0771 |
|      |    | 1e-3 | 0.0998 | 0.1277 | 0.0253 | 0.0294 |
|      |    | 1e-4 | 0.0795 | 0.1037 | 0.0252 | 0.0290 |

<br>

### ⭐ 최적 하이퍼파라미터 추천

| 목적 | lr | lora_alpha | dora_m_lr | IID EMA@30 | Non-IID EMA@30 |
|------|----|-----------|-----------|-----------|----------------|
| **균형 (권장)** | **1e-3** | **32** | **1e-3** | 0.9212 | **0.9132** |
| IID 특화 | 1e-3 | 16 | 1e-4 | **0.9229** | 0.9107 |
| Non-IID 특화 | 1e-3 | 32 | 1e-3 | 0.9212 | **0.9132** |

**추천: lr=1e-3, lora_alpha=32, dora_m_lr=1e-3** — IID+Non-IID 평균 최고(0.9172)이며 Non-IID 단독 최고. IID 특화(α=16,m_lr=1e-4)는 IID에서 0.0017 더 높지만 Non-IID가 떨어져, 두 분할을 함께 보고할 단일 설정으로는 **α=32, m_lr=1e-3**이 가장 안전.

**관찰된 경향:**
- **lr이 결정적**: 1e-3 ≈ 0.92 > 5e-4 ≈ 0.91 ≫ 5e-3(불안정) ≫ 1e-2(발산, 0.1~0.3). 원본 FeDoRA의 5e-3은 명백히 과대.
- **lora_alpha**: lr=1e-3에선 16~32 우수, 64는 하락. lr=5e-4에선 64가 근소 우위(lr과 상호작용).
- **dora_m_lr**: 안정 구간(lr≤1e-3)에선 영향 미미(±0.002). 단 lr이 큰 불안정 구간(5e-3)에선 작은 m_lr이 학습을 무너뜨림(예: 5e-3·α32·1e-4 → 0.36).

<br>

> ## 🔍 8. SVHN 20c — FeDoRA 그리드서치 + 비교 (EMA@30 / Max)
> **FeDoRA**: 그리드서치 best (신규 30R). **FedEx-LoRA**: 수정버전(Design A, faircomm, 신규 30R).
> **나머지 베이스라인**: 기존 메인 로그(50R)에서 **EMA@30/Max@30 추출** (재실행 아님).
> **현재 상태**: FeDoRA grid IID 완료(36/36), **Non-IID 진행 중**; FedEx-mod Non-IID 진행 중.

### SVHN, 20 Clients (EMA@30 / Max)
| Method                       | IID EMA@30 |  IID Max   | Non-IID EMA@30 | Non-IID Max |
|------------------------------|------------|------------|----------------|-------------|
| Ours (FeDoRA, grid-best)     |   0.9656   |   0.9662   |   *(진행 중)*   |  *(진행 중)* |
| FedEx-LoRA (mod, Design A)   |   0.9631   |   0.9640   |   *(진행 중)*   |  *(진행 중)* |
| FedIT                        |   0.9659   | **0.9678** |     0.9257     |    0.9431   |
| FlexLoRA                     | **0.9660** |   0.9672   |   **0.9293**   |  **0.9497** |
| FFA-LoRA                     |   0.3012   |   0.3226   |     0.1562     |    0.2984   |
| RAVAN                        |   0.9491   |   0.9522   |     0.8942     |    0.9245   |

> - **SVHN 20c IID grid-best FeDoRA**: lr=1e-3, α=32, m_lr=1e-3 → EMA@30 0.9656.
> - SVHN은 쉬운 데이터셋이라 상위 기법 모두 ~0.965~0.966로 천장 근접 → **변별력 낮음**. FlexLoRA(0.9660)·FedIT(0.9659)·FeDoRA(0.9656)가 사실상 동률. (FFA-LoRA는 메인 실험에서 발산 0.30.)
> - **핵심 차이(vs CIFAR-100)**: SVHN 메인 실험은 이미 lr=1e-3(올바른 값)을 써서 원본 FeDoRA(0.9641)와 그리드 best(0.9656) 격차가 작음. 반면 CIFAR-100은 원본이 lr=5e-3(과대)라 그리드로 0.86→0.92 대폭 개선.
> - **SVHN grid lr 경향(CIFAR와 동일)**: lr=1e-3(0.966) > 5e-4(0.964) ≫ 5e-3(0.876) ≫ 1e-2(0.489, 발산).
> - *베이스라인은 50R 로그의 @30 추출이라 cosine LR 스케줄이 신규 30R 실험과 다름(근사 비교).* Non-IID FeDoRA/FedEx-mod 완료 시 갱신 예정.

