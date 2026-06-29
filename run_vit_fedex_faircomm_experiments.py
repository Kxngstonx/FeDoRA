"""
Fair-Communication FedEx-LoRA 전체 ViT 환경 실험 스크립트
=========================================================
FedEx-LoRA 가 dense W0 를 매 라운드 재배포하여 발생하는 ~19배 다운링크 불공정을
제거한 변형. W0 는 사전학습값으로 동결하고, exact aggregation 의 누적 잔차를
rank-(res_rank) 저랭크 인자(res_A, res_B)로 유지/통신한다.

통신 예산 정합:
  다른 LoRA 계열(FedIT/FlexLoRA/FeDoRA) = rank 32 (lora_B/A)
  본 변형 = lora_r 16 (lora_B/A) + fedex_res_rank 16 (res_B/A) = 합산 rank 32
  → 라운드별 변경 통신량 O((16+16)(d+k)) = O(32(d+k)) 으로 '동일 예산'.

  ※ rank 를 낮추는 것만으로는 dense W0(=d×k, rank 무관) 비용이 줄지 않으므로,
    반드시 --fedex_lowrank_comm 으로 '전송 파라미터 자체'를 저랭크로 제한한다.

환경: run_vit_cifar100_svhn_experiments.py 와 동일
  (CIFAR-100 & SVHN × {20,50}c × {IID,Non-IID}, seed 42, lr 동일)

로그 method 이름: FedEx-LoRA-faircomm  (기존 결과 비파괴, 별도 저장)

사용법:
  python3 run_vit_fedex_faircomm_experiments.py                 # r=16, res=16 (예산 동일)
  python3 run_vit_fedex_faircomm_experiments.py --lora_r 32 --res_rank 32   # honest-2x 변형
  python3 run_vit_fedex_faircomm_experiments.py --datasets cifar100 --partitions noniid --dry-run
"""

import os
import time
import queue
import argparse
import threading
import subprocess


def wait_for_gpu_free_by_vram(gpu_id, vram_threshold_mb=5000, check_interval=30):
    first_wait = True
    while True:
        try:
            out = subprocess.check_output(
                f"nvidia-smi -i {gpu_id} --query-gpu=memory.used --format=csv,noheader,nounits",
                shell=True, text=True)
            used = int(out.strip())
            if used < vram_threshold_mb:
                if not first_wait:
                    print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} free ({used} MB). Next task...")
                break
            if first_wait:
                print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} BUSY ({used} MB). Waiting...")
                first_wait = False
            time.sleep(check_interval)
        except subprocess.CalledProcessError:
            time.sleep(check_interval)


def run_command(command, gpu_id, task_name, log_dir):
    print(f"[{time.strftime('%H:%M:%S')}] Starting '{task_name}' on GPU {gpu_id}...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    log_path = os.path.join(log_dir, f"{task_name}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.Popen(command, shell=True, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            f.write(line)
            f.flush()
        proc.wait()
    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] Finished '{task_name}' on GPU {gpu_id} [{status}]")
    print("-" * 60)


def worker(gpu_id, task_queue):
    while True:
        try:
            task_name, command, log_dir = task_queue.get_nowait()
        except queue.Empty:
            break
        wait_for_gpu_free_by_vram(gpu_id)
        run_command(command, gpu_id, task_name, log_dir)
        task_queue.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--lora_r", type=int, default=16, help="LoRA 어댑터 rank (lora_B/A)")
    ap.add_argument("--res_rank", type=int, default=16, help="저랭크 잔차 어댑터 rank (res_B/A)")
    ap.add_argument("--datasets", nargs="+", default=["cifar100", "svhn"])
    ap.add_argument("--clients", type=int, nargs="+", default=[20, 50])
    ap.add_argument("--partitions", nargs="+", default=["iid", "noniid"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model = "vit-base"
    rounds = 50
    local_steps = 50
    batch_size = 32
    seed = 42
    beta = 0.3
    sample_fraction = {20: 0.15, 50: 0.06}

    total_rank = args.lora_r + args.res_rank
    method_name = f"FedEx-LoRA-faircomm-r{args.lora_r}res{args.res_rank}"
    method_args = (
        f"--peft lora --fedex_lora --fedex_lowrank_comm --fedex_res_rank {args.res_rank} "
        f"--trainable_A --lora_r {args.lora_r} --lora_alpha {args.lora_r} --target_modules qkv"
    )
    lr_table = {
        'cifar100': {'iid': '1e-3', 'noniid': '1e-3'},
        'svhn':     {'iid': '1e-3', 'noniid': '1e-3'},
    }

    log_base_dir = "./logs/vit_cifar100_svhn_experiments"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    task_queue = queue.Queue()
    planned = []

    for dataset in args.datasets:
        for n_clients in args.clients:
            sf = sample_fraction[n_clients]
            for partition in args.partitions:
                lr = lr_table[dataset][partition]
                task_name = f"{dataset}_{n_clients}c_{partition}_lower_{method_name}_seed{seed}"
                log_dir = os.path.join(log_base_dir, f"{dataset}_lower")
                os.makedirs(log_dir, exist_ok=True)

                common = (
                    f"--model {model} --dataset {dataset} --datadir /mnt/data1 --round {rounds} "
                    f"--epochs 1 --n_clients {n_clients} --sample_fraction {sf} "
                    f"--seed {seed} --optimizer adam --lr {lr} --reg 0 "
                    f"--scheduler cosine --local_steps {local_steps} --batch_size {batch_size} "
                )
                common += (f"--partition noniid --beta {beta} " if partition == "noniid"
                           else "--partition iid ")

                cmd = (
                    f"cd {script_dir} && python3 main.py "
                    f"{common} {method_args} --ft_classifier "
                    f"--logdir {log_dir} --log_file_name '{task_name}'"
                )
                task_queue.put((task_name, cmd, log_dir))
                planned.append(task_name)

    print(f"[{time.strftime('%H:%M:%S')}] Fair-comm FedEx-LoRA "
          f"(lora_r={args.lora_r} + res_rank={args.res_rank} = total rank {total_rank}) "
          f"— {len(planned)} runs across {len(args.gpus)} GPUs")
    print(f"  per-round changing comm ≈ O({total_rank}·(d+k)) (다른 LoRA 계열과 동일 예산)")
    for t in planned:
        print(f"   {t}")
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN — not executing.")
        return

    threads = []
    for gpu_id in args.gpus:
        th = threading.Thread(target=worker, args=(gpu_id, task_queue))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    print(f"[{time.strftime('%H:%M:%S')}] All fair-comm FedEx-LoRA experiments completed.")


if __name__ == "__main__":
    main()
