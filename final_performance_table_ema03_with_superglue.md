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

| Method                    |    MNLI    |    QQP     |    SST2    |
|---------------------------|------------|------------|------------|
| **🌟 Ours (FeDoRA)**       |   0.8377   |   0.8316   | **0.9456** |
| FL+DoRA(FlexLoRA)         | **0.8381** | **0.8378** |   0.9400   |
| FedEx-LoRA                |   0.8332   |   0.8377   |   0.9443   |
| FedIT                     |   0.8294   |   0.8340   |   0.9443   |
| FlexLoRA                  |   0.8061   |   0.8301   |   0.9436   |
| FL+DoRA(FlexLoRA+FFALoRA) |   0.7915   |   0.8070   |   0.9421   |
| FFA-LoRA                  |   0.7829   |   0.8061   |   0.9424   |

<br>

## 📚 2. SuperGLUE Benchmark (Multi-Seed Analysis)

### 🌱 Seed 42 (기존 결과 - 'Lucky' Partition)
| Method                    |   BOOLQ    |  MULTIRC   |   RECORD   |    WIC     |
|---------------------------|------------|------------|------------|------------|
| **🌟 Ours (FeDoRA)**       |   0.7986   |   0.5716   |     -      |   0.6911   |
| FL+DoRA(FlexLoRA)         |   0.8048   | **0.5729** | **0.8675** |   0.6967   |
| FedEx-LoRA                | **0.8050** |   0.5705   |     -      | **0.6987** |
| FedIT                     |   0.8040   |   0.5721   |   0.8660   |   0.6908   |
| FlexLoRA                  |   0.7977   |   0.5721   |     -      |   0.6939   |
| FL+DoRA(FlexLoRA+FFALoRA) |   0.7618   |   0.5720   |     -      |   0.6706   |
| FFA-LoRA                  |   0.7624   |   0.5720   |     -      |   0.6790   |

### 🌱 Seed 43 (Extreme Non-IID Partition)
| Method                    |   BOOLQ    |  MULTIRC   |   RECORD   |    WIC     |
|---------------------------|------------|------------|------------|------------|
| **🌟 Ours (FeDoRA)**       | **0.6219** |   0.5630   | **0.8660** |     -      |
| FL+DoRA(FlexLoRA)         |     -      |     -      |     -      |     -      |
| FedEx-LoRA                |   0.6218   |   0.5685   | **0.8660** |     -      |
| FedIT                     |   0.6217   |   0.5635   | **0.8660** |     -      |
| FlexLoRA                  |   0.6217   | **0.5745** | **0.8660** |     -      |
| FL+DoRA(FlexLoRA+FFALoRA) |     -      |     -      |     -      |     -      |
| FFA-LoRA                  |   0.6217   |   0.5718   | **0.8660** |     -      |

### 🌱 Seed 44 (Extreme Non-IID Partition)
| Method                    |   BOOLQ    |  MULTIRC   |   RECORD   |    WIC     |
|---------------------------|------------|------------|------------|------------|
| **🌟 Ours (FeDoRA)**       |   0.6738   |   0.5356   |   0.8684   |     -      |
| FL+DoRA(FlexLoRA)         |     -      |     -      |     -      |     -      |
| FedEx-LoRA                |   0.6848   |   0.5674   |   0.8673   |     -      |
| FedIT                     |   0.6871   |   0.5566   |   0.8686   |     -      |
| FlexLoRA                  | **0.6992** | **0.5687** | **0.8699** |     -      |
| FL+DoRA(FlexLoRA+FFALoRA) |     -      |     -      |     -      |     -      |
| FFA-LoRA                  |   0.6429   |   0.5483   |   0.8660   |     -      |

<br>

## 📚 3. ANLI (Adversarial NLI)

| Method                    | ANLI (Avg) |     R1     |     R2     |     R3     |
|---------------------------|------------|------------|------------|------------|
| **🌟 Ours (FeDoRA)**       |   0.4294   |   0.5396   | **0.3745** | **0.3741** |
| FL+DoRA(FlexLoRA)         | **0.4331** | **0.5566** |   0.3742   |   0.3684   |
| FedEx-LoRA                |   0.4038   |   0.4980   |   0.3570   |   0.3564   |
| FedIT                     |   0.4097   |   0.5134   |   0.3531   |   0.3625   |
| FlexLoRA                  |   0.4162   |   0.5249   |   0.3648   |   0.3588   |
| FL+DoRA(FlexLoRA+FFALoRA) |   0.3614   |   0.4090   |   0.3405   |   0.3347   |
| FFA-LoRA                  |   0.3495   |   0.3603   |   0.3498   |   0.3385   |

<br>

## 📚 4. MMLU Benchmark (Massive Multitask Language Understanding)

| Method                    |    MMLU    |
|---------------------------|------------|
| **🌟 Ours (FeDoRA)**       |   0.2531   |
| FL+DoRA(FlexLoRA)         |     -      |
| FedEx-LoRA                |   0.2546   |
| FedIT                     |   0.2524   |
| FlexLoRA                  |   0.2536   |
| FL+DoRA(FlexLoRA+FFALoRA) |     -      |
| FFA-LoRA                  | **0.2549** |

<br>

## 📚 5. Vision Benchmarks (CIFAR-100 & SVHN)
*Note: Results are based on the currently executed experiments (Lower Budget, Seed 42). Experiments marked with an asterisk (*) are still in progress.*

### 20 Clients Setting
| Method                    | CIFAR-100 (IID) | CIFAR-100 (Non-IID) | SVHN (IID) | SVHN (Non-IID) |
|---------------------------|-----------------|---------------------|------------|----------------|
| **🌟 Ours (FeDoRA)**       |          0.8320 |              0.8113 |     0.5520 |         0.5050 |
| FedEx-LoRA                |          0.8311 |              0.8141 |     0.5520 |         0.5050 |
| FedIT                     |          0.8320 |              0.8113 |     0.5520 |         0.5050 |
| FlexLoRA                  |          0.8320 |              0.8113 |     0.5520 |         0.5050 |
| FFA-LoRA                  |          0.8311 |              0.8141 |     0.5520 |         0.5050 |
| RAVAN                     |          0.7826 |              0.7314 |     0.4386 |         0.3425 |

### 50 Clients Setting
| Method                    | CIFAR-100 (IID) | CIFAR-100 (Non-IID) | SVHN (IID) | SVHN (Non-IID) |
|---------------------------|-----------------|---------------------|------------|----------------|
| **🌟 Ours (FeDoRA)**       |          0.8278 |              0.8098 |          - |              - |
| FedEx-LoRA                |          0.8244 |              0.8056 |     0.5464 |              - |
| FedIT                     |          0.8278 |              0.8098 |     0.5464 |              - |
| FlexLoRA                  |          0.8278 |              0.8098 |          - |              - |
| FFA-LoRA                  |          0.8244 |              0.8056 |   *0.4898* |              - |
| RAVAN                     |          0.7526 |              0.6811 |          - |              - |
