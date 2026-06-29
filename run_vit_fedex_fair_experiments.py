"""
Fair FedEx-LoRA 재실험 스크립트 (W0 동결 버전)
================================================
기존 run_vit_cifar100_svhn_experiments.py 와 '완전히 동일한' 실험 환경
(ViT-B/16, Lower Budget, CIFAR-100 & SVHN × {20,50} clients × {IID, Non-IID}, seed 42)에서
FedEx-LoRA 만 재실행한다.

차이점은 단 하나: --fedex_freeze_w0 플래그를 추가하여 사전학습 가중치 W0를
서버 집계 시 갱신하지 않는다(잔차 write-back 생략).

  - 기존(불공정) FedEx-LoRA: 매 라운드 W0 += residual → base model이 full-rank delta를
    누적하고, state_dict 배포 시 갱신된 W0(O(d^2))가 클라이언트로 재전송됨.
  - 본 스크립트(공정) FedEx-LoRA: W0를 사전학습값으로 동결 → 다른 PEFT 베이스라인
    (FeDoRA/FlexLoRA/FedIT/FFA-LoRA)과 동일하게 O(r) 통신·frozen base 조건을 만족.

결과 로그는 기존 결과를 덮어쓰지 않도록 method 이름에 '-fairW0' 접미사를 붙여
동일 디렉토리(logs/vit_cifar100_svhn_experiments/{dataset}_lower)에 저장한다.
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
    # 설정 — 기존 run_vit_cifar100_svhn_experiments.py 와 동일
    # =============================================
    gpus = [0, 1]
    seeds = [42]

    datasets = ["cifar100", "svhn"]
    n_clients_list = [20, 50]
    partitions = ["iid", "noniid"]

    model = "vit-base"
    rounds = 50
    local_steps = 50
    batch_size = 32
    sample_fraction_3_of_20 = 0.15   # 3/20
    sample_fraction_3_of_50 = 0.06   # 3/50
    beta = 0.3  # Dirichlet alpha for Non-IID

    # =============================================
    # FedEx-LoRA (Fair, W0-frozen) — 원본과 동일한 인자에 --fedex_freeze_w0 추가
    # 원본: '--peft lora --fedex_lora --trainable_A --lora_r 32 --lora_alpha 32 --target_modules qkv'
    # =============================================
    method_name = "FedEx-LoRA-fairW0"
    method_args = (
        "--peft lora --fedex_lora --fedex_freeze_w0 --trainable_A "
        "--lora_r 32 --lora_alpha 32 --target_modules qkv"
    )
    # lr: 원본 FedEx-LoRA 와 동일하게 모든 환경에서 1e-3
    lr_table = {
        'cifar100': {'iid': '1e-3', 'noniid': '1e-3'},
        'svhn':     {'iid': '1e-3', 'noniid': '1e-3'},
    }

    log_base_dir = "./logs/vit_cifar100_svhn_experiments"
    task_queue = queue.Queue()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for dataset in datasets:
        for n_clients in n_clients_list:
            sample_fraction = sample_fraction_3_of_20 if n_clients == 20 else sample_fraction_3_of_50
            for partition in partitions:
                lr = lr_table[dataset][partition]
                for seed in seeds:
                    task_name = f"{dataset}_{n_clients}c_{partition}_lower_{method_name}_seed{seed}"
                    log_dir = os.path.join(log_base_dir, f"{dataset}_lower")
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

                    classifier_arg = "--ft_classifier"

                    cmd = (
                        f"cd {script_dir} && python3 main.py "
                        f"{common_args} {method_args} {classifier_arg} "
                        f"--logdir {log_dir} --log_file_name '{task_name}'"
                    )
                    task_queue.put((task_name, cmd, log_dir))

    total_tasks = task_queue.qsize()
    print(f"[{time.strftime('%H:%M:%S')}] Starting {total_tasks} FAIR FedEx-LoRA (W0-frozen) experiments across {len(gpus)} GPUs")
    print(f"Datasets: {datasets} | Clients: {n_clients_list} | Partitions: {partitions} | Seeds: {seeds}")
    print(f"Method args: {method_args}")
    print("=" * 60)

    threads = []
    for gpu_id in gpus:
        t = threading.Thread(target=worker, args=(gpu_id, task_queue))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"[{time.strftime('%H:%M:%S')}] All {total_tasks} fair FedEx-LoRA experiments have been completed.")


if __name__ == "__main__":
    main()
