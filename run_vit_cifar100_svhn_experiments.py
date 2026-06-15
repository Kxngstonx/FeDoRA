"""
RAVAN Table 2 재현 실험 스크립트
=================================
ViT-B/16 모델로 CIFAR-100 & SVHN에서 6개 방법을 비교합니다.
방법: FeDoRA, FedEx-LoRA, FedIT, FlexLoRA, FFA-LoRA, RAVAN

논문 설정:
- |C| = 20 또는 50 clients
- 라운드당 3 clients 참여 (uniform random)
- Dirichlet α=0.3 (Non-IID) / 균등 분할 (IID)
- 50 local steps (mini-batch 단위)
- 50 communication rounds
- Adam optimizer (momentum=0.9)
- LoRA target: query, value (ViT self-attention)
- 3 random seeds 평균
"""

import os
import subprocess
import time
import queue
import threading


def wait_for_gpu_free_by_vram(gpu_id, vram_threshold_mb=5000, check_interval=30):
    """지정된 GPU의 VRAM 사용량이 임계값 이하로 떨어질 때까지 대기합니다."""
    print(f"[{time.strftime('%H:%M:%S')}] Checking if GPU {gpu_id} VRAM is below {vram_threshold_mb} MB...")
    first_wait = True
    while True:
        try:
            output = subprocess.check_output(
                f"nvidia-smi -i {gpu_id} --query-gpu=memory.used --format=csv,noheader,nounits",
                shell=True, text=True
            )
            vram_used = int(output.strip())
            if vram_used < vram_threshold_mb:
                if not first_wait:
                    print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} VRAM ({vram_used} MB) is now below {vram_threshold_mb} MB. Starting next task...")
                    print("=" * 60)
                break
            if first_wait:
                print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} is currently BUSY (VRAM: {vram_used} MB). Waiting...")
                first_wait = False
            time.sleep(check_interval)
        except subprocess.CalledProcessError:
            print(f"[{time.strftime('%H:%M:%S')}] Failed to get nvidia-smi for GPU {gpu_id}. Retrying later.")
            time.sleep(check_interval)


def run_command(command, gpu_id, task_name, log_dir):
    """지정된 GPU에서 단일 명령어를 실행하고 로그를 파일에 기록합니다."""
    print(f"[{time.strftime('%H:%M:%S')}] Starting '{task_name}' on GPU {gpu_id}...\n")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    log_file_path = os.path.join(log_dir, f"{task_name}.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "w") as f:
        process = subprocess.Popen(
            command, shell=True, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in process.stdout:
            print(f"[{task_name}|GPU{gpu_id}] {line}", end="")
            f.write(line)
            f.flush()
        process.wait()
    if process.returncode == 0:
        print(f"[{time.strftime('%H:%M:%S')}] Successfully finished '{task_name}' on GPU {gpu_id}.")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Task '{task_name}' failed on GPU {gpu_id} with return code {process.returncode}.")
    print("-" * 60)


def worker(gpu_id, task_queue):
    while True:
        try:
            task_name, command, log_dir = task_queue.get_nowait()
        except queue.Empty:
            break
        wait_for_gpu_free_by_vram(gpu_id, vram_threshold_mb=5000, check_interval=30)
        run_command(command, gpu_id, task_name, log_dir)
        task_queue.task_done()


def main():
    # =============================================
    # 설정
    # =============================================
    gpus = [0, 1]
    seeds = [42]  # Default to 1 seed as requested

    datasets = ["cifar100", "svhn"]
    n_clients_list = [20, 50]
    partitions = ["iid", "noniid"]

    model = "vit-base"
    rounds = 50
    local_steps = 50
    batch_size = 32
    sample_fraction_3_of_20 = 0.15   # 3/20 = 0.15
    sample_fraction_3_of_50 = 0.06   # 3/50 = 0.06
    beta = 0.3  # Dirichlet alpha for Non-IID

    # =============================================
    # Method 정의 — Lower Parameter Budget (N_total ≈ 1.2M)
    # =============================================
    # ViT-B/16 (timm) uses FUSED qkv layer: qkv (768→2304) per block, 12 blocks
    # target_modules = 'qkv' (NOT 'query value' - timm ViT doesn't have separate q/k/v)
    # FedIT:      r=32 → N = (768+2304) * 32 * 12 ≈ 1.18M (trainable A+B)
    # FedEx-LoRA: r=32 → same
    # FFA-LoRA:   r=64 → N = 2304 * 64 * 12 ≈ 1.77M (only B trainable, double rank)
    # FlexLoRA:   r=32 → same as FedIT (trainable A+B, SVD aggregation)
    # FeDoRA:     r=32 → same as FlexLoRA (DoRA with SVD-A aggregation)
    # RAVAN:      r=110, 4 heads → N = 4 * 110^2 * 12 ≈ 580K (only H_i trainable)

    METHODS_LOWER = {
        'FedIT': {
            'args': '--peft lora --trainable_A --lora_r 32 --lora_alpha 32 --target_modules qkv',
            'lr': {
                'cifar100': {'iid': '5e-3', 'noniid': '5e-3'},
                'svhn':     {'iid': '1e-3', 'noniid': '1e-3'},
            }
        },
        'FedEx-LoRA': {
            'args': '--peft lora --fedex_lora --trainable_A --lora_r 32 --lora_alpha 32 --target_modules qkv',
            'lr': {
                'cifar100': {'iid': '1e-3', 'noniid': '1e-3'},
                'svhn':     {'iid': '1e-3', 'noniid': '1e-3'},
            }
        },
        'FFA-LoRA': {
            'args': '--peft lora --flex_lora --flex_lora_freeze_a --lora_r 64 --lora_alpha 64 --target_modules qkv',
            'lr': {
                'cifar100': {'iid': '1e-2', 'noniid': '1e-2'},
                'svhn':     {'iid': '1e-2', 'noniid': '1e-2'},
            }
        },
        'FlexLoRA': {
            'args': '--peft lora --flex_lora --lora_r 32 --lora_alpha 32 --target_modules qkv',
            'lr': {
                'cifar100': {'iid': '5e-3', 'noniid': '5e-3'},
                'svhn':     {'iid': '1e-3', 'noniid': '1e-3'},
            }
        },
        'FeDoRA': {
            'args': '--peft dora --flex_lora --flex_lora_svd_a --lora_r 32 --lora_alpha 32 --target_modules qkv',
            'lr': {
                'cifar100': {'iid': '5e-3', 'noniid': '5e-3'},
                'svhn':     {'iid': '1e-3', 'noniid': '1e-3'},
            }
        },
        'RAVAN': {
            'args': '--peft ravan --lora_r 110 --lora_alpha 110 --ravan_heads 4 --ravan_init gram_schmidt --target_modules qkv',
            'lr': {
                'cifar100': {'iid': '5e-4', 'noniid': '5e-4'},
                'svhn':     {'iid': '5e-4', 'noniid': '5e-4'},
            }
        },
    }

    log_base_dir = "./logs/vit_cifar100_svhn_experiments"
    task_queue = queue.Queue()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # =============================================
    # 태스크 생성
    # =============================================
    budget_configs = [
        ('lower', METHODS_LOWER),
    ]

    for budget_name, methods in budget_configs:
        for dataset in datasets:
            for n_clients in n_clients_list:
                sample_fraction = sample_fraction_3_of_20 if n_clients == 20 else sample_fraction_3_of_50
                for partition in partitions:
                    for method_name, method_cfg in methods.items():
                        lr = method_cfg['lr'][dataset][partition]
                        method_args = method_cfg['args']

                        for seed in seeds:
                            task_name = f"{dataset}_{n_clients}c_{partition}_{budget_name}_{method_name}_seed{seed}"
                            log_dir = os.path.join(log_base_dir, f"{dataset}_{budget_name}")
                            os.makedirs(log_dir, exist_ok=True)

                            common_args = (
                                f"--model {model} --dataset {dataset} --datadir /mnt/data1 --round {rounds} "
                                f"--epochs 1 --n_clients {n_clients} --sample_fraction {sample_fraction} "
                                f"--seed {seed} --optimizer adam --lr {lr} --reg 0 "
                                f"--scheduler cosine --local_steps {local_steps} --batch_size {batch_size} "
                            )

                            if partition == 'noniid':
                                common_args += f"--partition noniid --beta {beta} "
                            else:
                                common_args += "--partition iid "

                            # ViT classifier head: full fine-tune (not LoRA)
                            classifier_arg = "--ft_classifier"

                            cmd = (
                                f"cd {script_dir} && python3 main.py "
                                f"{common_args} {method_args} {classifier_arg} "
                                f"--logdir {log_dir} --log_file_name '{task_name}'"
                            )
                            task_queue.put((task_name, cmd, log_dir))

    total_tasks = task_queue.qsize()
    print(f"[{time.strftime('%H:%M:%S')}] Starting {total_tasks} ViT CIFAR-100/SVHN experiments across {len(gpus)} GPUs")
    print(f"Datasets: {datasets}")
    print(f"Clients: {n_clients_list}")
    print(f"Partitions: {partitions}")
    print(f"Budgets: lower, higher")
    print(f"Methods: {list(METHODS_LOWER.keys())}")
    print(f"Seeds: {seeds}")
    print("=" * 60)

    # =============================================
    # 워커 스레드 시작
    # =============================================
    threads = []
    for gpu_id in gpus:
        t = threading.Thread(target=worker, args=(gpu_id, task_queue))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"[{time.strftime('%H:%M:%S')}] All {total_tasks} experiments have been completed.")


if __name__ == "__main__":
    main()
