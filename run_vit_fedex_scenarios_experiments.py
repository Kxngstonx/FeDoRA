"""
FedEx-LoRA 시나리오 비교 실험 (전체 ViT 환경)
==============================================
동일한 ViT 실험 환경(run_vit_cifar100_svhn_experiments.py 와 같음:
CIFAR-100 & SVHN × {20,50}c × {IID,Non-IID}, seed 42)에서 FedEx-LoRA 의
여러 '통신 시나리오'를 모두 돌려 한 표로 비교할 수 있게 한다.

시나리오 (--scenarios 로 선택, 기본 전체):
  orig         원본 FedEx-LoRA. dense W0 를 매 라운드 누적·재배포 (~19× 다운링크).
               [이미 메인 표에 있음 — 재현/검증용]
  fairW0       --fedex_freeze_w0 : W0 동결(잔차 write-back 생략). FedIT 와 등가, O(r) 통신.
  faircomm16   --fedex_lowrank_comm --fedex_res_rank 16, lora_r 16 (합산 rank 32):
               누적 잔차를 rank-16 저랭크 어댑터로 통신. 다른 LoRA 계열과 '동일 예산'(Design A).
  faircomm32   --fedex_lowrank_comm --fedex_res_rank 32, lora_r 32:
               honest-2x 변형 (잔차 rank 32). 통신 ~2×, 절단 손실 최소.

로그 method 이름:
  FedEx-LoRA-<scenario>  (예: FedEx-LoRA-faircomm16) — 기존 결과 비파괴, 별도 저장.

★ 주의: faircomm* 시나리오는 새로 추가한 모델 파라미터(res_A/res_B)·집계 경로를
  사용한다. 전체 실행 전 반드시 --smoke (1개 환경 × 3라운드)로 정상 동작을 확인할 것.

사용법:
  python3 run_vit_fedex_scenarios_experiments.py --smoke                 # 빠른 동작 확인(3R)
  python3 run_vit_fedex_scenarios_experiments.py                         # 전체 시나리오×8환경
  python3 run_vit_fedex_scenarios_experiments.py --scenarios faircomm16  # 특정 시나리오만
  python3 run_vit_fedex_scenarios_experiments.py --dry-run
"""

import os
import time
import queue
import argparse
import threading
import subprocess

SCENARIOS = {
    # name        : (extra_args, lora_r, lora_alpha)
    "orig":        ("",                                                32, 32),
    "fairW0":      ("--fedex_freeze_w0",                               32, 32),
    "faircomm16":  ("--fedex_lowrank_comm --fedex_res_rank 16",        16, 16),
    "faircomm32":  ("--fedex_lowrank_comm --fedex_res_rank 32",        32, 32),
}


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
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS.keys()),
                    choices=list(SCENARIOS.keys()))
    ap.add_argument("--datasets", nargs="+", default=["cifar100", "svhn"])
    ap.add_argument("--clients", type=int, nargs="+", default=[20, 50])
    ap.add_argument("--partitions", nargs="+", default=["iid", "noniid"])
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--smoke", action="store_true",
                    help="빠른 동작 확인: cifar100 20c noniid 1개 환경 × 3라운드, faircomm16 시나리오")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 고정 환경 (run_vit_cifar100_svhn_experiments.py 와 동일)
    model = "vit-base"
    local_steps = 50
    batch_size = 32
    seed = 42
    beta = 0.3
    sample_fraction = {20: 0.15, 50: 0.06}
    lr = "1e-3"   # FedEx-LoRA 는 전 환경 1e-3

    if args.smoke:
        args.scenarios = ["faircomm16"]
        args.datasets = ["cifar100"]; args.clients = [20]; args.partitions = ["noniid"]
        args.rounds = 3
        print(">> SMOKE TEST: faircomm16 / cifar100_20c_noniid / 3 rounds")

    log_base_dir = "./logs/vit_fedex_scenarios"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    task_queue = queue.Queue()
    planned = []

    for scen in args.scenarios:
        extra, lr_, la_ = SCENARIOS[scen]
        for dataset in args.datasets:
            for n_clients in args.clients:
                sf = sample_fraction[n_clients]
                for partition in args.partitions:
                    tag = f"{dataset}_{n_clients}c_{partition}_lower_FedEx-LoRA-{scen}_seed{seed}"
                    if args.smoke:
                        tag = "SMOKE_" + tag
                    log_dir = os.path.join(log_base_dir, f"{dataset}_lower")
                    os.makedirs(log_dir, exist_ok=True)

                    method_args = (
                        f"--peft lora --fedex_lora --trainable_A "
                        f"--lora_r {lr_} --lora_alpha {la_} --target_modules qkv {extra}"
                    )
                    common = (
                        f"--model {model} --dataset {dataset} --datadir /mnt/data1 --round {args.rounds} "
                        f"--epochs 1 --n_clients {n_clients} --sample_fraction {sf} "
                        f"--seed {seed} --optimizer adam --lr {lr} --reg 0 "
                        f"--scheduler cosine --local_steps {local_steps} --batch_size {batch_size} "
                    )
                    common += (f"--partition noniid --beta {beta} " if partition == "noniid"
                               else "--partition iid ")
                    cmd = (
                        f"cd {script_dir} && python3 main.py "
                        f"{common} {method_args} --ft_classifier "
                        f"--logdir {log_dir} --log_file_name '{tag}'"
                    )
                    task_queue.put((tag, cmd, log_dir))
                    planned.append(tag)

    print(f"[{time.strftime('%H:%M:%S')}] FedEx scenarios — {len(planned)} runs "
          f"(scenarios={args.scenarios}) across {len(args.gpus)} GPUs, {args.rounds} rounds")
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
    print(f"[{time.strftime('%H:%M:%S')}] All FedEx scenario experiments completed. "
          f"Run parse_fedex_comparison.py to build the comparison table.")


if __name__ == "__main__":
    main()
