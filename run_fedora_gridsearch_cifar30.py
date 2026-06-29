"""
FeDoRA 그리드서치 — CIFAR-100, 30 rounds, 72 runs (FeDoRA 전용, 기법 불변)
========================================================================
지시 사양:
  - 데이터셋: CIFAR-100 (20 clients, lower budget, seed 42)
  - 분할: IID + Non-IID(Dirichlet beta=0.3)
  - 그리드: dora_m_lr(3) × lr(4) × lora_alpha(3) = 36 조합 / partition
            × 2 partitions = 72 runs
  - 라운드: 30 (조기 종료로 탐색 가속)
  - 기법 불변: --peft dora --flex_lora --flex_lora_svd_a (warm-up 미사용)

하이퍼파라미터 후보군 (관련 논문 사용값 기반):
  lr ∈ {5e-4, 1e-3, 5e-3, 1e-2}
     - LoRA(Hu et al., 2021), DoRA(Liu et al., 2024), FedIT/Shepherd, RAVAN(2025)에서
       ViT/LoRA 파인튜닝에 통용되는 로그스케일 범위. (현 FeDoRA-CIFAR 기본값 5e-3 포함)
  lora_alpha ∈ {16, 32, 64}  (r=32 → scaling alpha/r = 0.5, 1, 2)
     - LoRA 원논문 alpha=2r, DoRA alpha=r 관행을 모두 포함.
  dora_m_lr ∈ {same, 1e-3, 1e-4}
     - DoRA 기본은 방향과 동일 lr(=same). 일부 FL 연구는 magnitude를 느리게 갱신 → 1e-3, 1e-4 포함.

로그: logs/fedora_gridsearch_cifar30/cifar100/{run_tag}.log  (메인 실험 비파괴)
요약: python3 run_fedora_gridsearch_cifar30.py --summarize   (EMA 0.3 @ Round 30)
"""

import os
import re
import sys
import time
import queue
import argparse
import itertools
import threading
import subprocess

# ---------------- 고정 (기법/환경) ----------------
FEDORA_FIXED_ARGS = "--peft dora --flex_lora --flex_lora_svd_a --target_modules qkv"
MODEL = "vit-base"
DATASET = "cifar100"
N_CLIENTS = 20
SAMPLE_FRACTION = 0.15          # 3/20
ROUNDS = 30
EPOCHS = 1
LOCAL_STEPS = 50
BATCH_SIZE = 32
SEED = 42
DATADIR = "/mnt/data1"
BETA = 0.3
LORA_R = 32
EMA_ALPHA = 0.7
EMA_TARGET_ROUND = 30
LOG_DIR = "./logs/fedora_gridsearch_cifar30/cifar100"

# ---------------- 그리드 (논문 기반 후보군) ----------------
GRID = {
    "lr":         ["5e-4", "1e-3", "5e-3", "1e-2"],   # 4
    "lora_alpha": [16, 32, 64],                       # 3
    "dora_m_lr":  ["same", "1e-3", "1e-4"],           # 3
}
PARTITIONS = ["iid", "noniid"]                        # × 2  → 4×3×3×2 = 72


def build_tag(part, hp):
    def s(x):
        return str(x).replace(".", "p").replace("-", "m")
    return (f"fedora_{DATASET}_{N_CLIENTS}c_{part}_lr{s(hp['lr'])}"
            f"_a{hp['lora_alpha']}_mlr{s(hp['dora_m_lr'])}_r{LORA_R}_R{ROUNDS}_seed{SEED}")


def build_cmd(part, hp, script_dir, tag):
    common = (
        f"--model {MODEL} --dataset {DATASET} --datadir {DATADIR} --round {ROUNDS} "
        f"--epochs {EPOCHS} --n_clients {N_CLIENTS} --sample_fraction {SAMPLE_FRACTION} "
        f"--seed {SEED} --optimizer adam --lr {hp['lr']} --reg 0 "
        f"--scheduler cosine --local_steps {LOCAL_STEPS} --batch_size {BATCH_SIZE} "
    )
    common += (f"--partition noniid --beta {BETA} " if part == "noniid" else "--partition iid ")
    tune = f"--lora_r {LORA_R} --lora_alpha {hp['lora_alpha']} "
    if hp["dora_m_lr"] != "same":
        tune += f"--dora_m_lr {hp['dora_m_lr']} "
    # sys.executable: 이 러너를 띄운 venv python 을 서브프로세스에도 그대로 사용
    return (f"cd {script_dir} && {sys.executable} main.py "
            f"{common} {FEDORA_FIXED_ARGS} {tune} --ft_classifier "
            f"--logdir {LOG_DIR} --log_file_name '{tag}'")


def enumerate_tasks(script_dir):
    tasks = []
    keys = list(GRID.keys())
    for part in PARTITIONS:
        for combo in itertools.product(*[GRID[k] for k in keys]):
            hp = dict(zip(keys, combo))
            tag = build_tag(part, hp)
            tasks.append((tag, build_cmd(part, hp, script_dir, tag)))
    return tasks


# ---------------- GPU 워커 ----------------
def wait_for_gpu_free_by_vram(gpu_id, vram_threshold_mb=5000, check_interval=30):
    first = True
    while True:
        try:
            used = int(subprocess.check_output(
                f"nvidia-smi -i {gpu_id} --query-gpu=memory.used --format=csv,noheader,nounits",
                shell=True, text=True).strip())
            if used < vram_threshold_mb:
                if not first:
                    print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} free ({used} MB).", flush=True)
                break
            if first:
                print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} BUSY ({used} MB). Waiting...", flush=True)
                first = False
            time.sleep(check_interval)
        except subprocess.CalledProcessError:
            time.sleep(check_interval)


def run_command(command, gpu_id, tag):
    print(f"[{time.strftime('%H:%M:%S')}] START '{tag}' on GPU {gpu_id}", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    os.makedirs(LOG_DIR, exist_ok=True)
    lp = os.path.join(LOG_DIR, f"{tag}.log")
    with open(lp, "w") as f:
        p = subprocess.Popen(command, shell=True, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line); f.flush()
        p.wait()
    print(f"[{time.strftime('%H:%M:%S')}] DONE '{tag}' [{'OK' if p.returncode==0 else 'FAIL rc='+str(p.returncode)}]", flush=True)


def worker(gpu_id, task_queue):
    while True:
        try:
            tag, cmd = task_queue.get_nowait()
        except queue.Empty:
            break
        wait_for_gpu_free_by_vram(gpu_id)
        run_command(cmd, gpu_id, tag)
        task_queue.task_done()


# ---------------- 요약 ----------------
def get_accs(p):
    accs, seen = [], set()
    pat = re.compile(r'(\d{2}-\d{2} \d{2}:\d{2}).*>> Global Model Test accuracy: ([0-9.]+)')
    if not os.path.exists(p):
        return accs
    for line in open(p, errors='ignore'):
        m = pat.search(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1)); accs.append(float(m.group(2)))
    return accs


def ema(accs, al=EMA_ALPHA, t=EMA_TARGET_ROUND):
    if not accs:
        return None
    e = accs[0]; vals = [e]
    for v in accs[1:]:
        e = al * e + (1 - al) * v; vals.append(e)
    return vals[min(t - 1, len(vals) - 1)]


def summarize():
    if not os.path.isdir(LOG_DIR):
        print("no logs yet"); return
    pat = re.compile(r"fedora_cifar100_\d+c_(?P<part>iid|noniid)_(?P<rest>.+)")
    rows = []
    for fn in os.listdir(LOG_DIR):
        if not fn.endswith(".log"):
            continue
        m = pat.match(fn[:-4])
        if not m:
            continue
        a = get_accs(os.path.join(LOG_DIR, fn))
        rows.append({"part": m["part"], "cfg": m["rest"], "n": len(a), "ema": ema(a)})
    done = [r for r in rows if r["ema"] is not None]
    if not done:
        print("no completed runs"); return
    print("=" * 78)
    print(f"FeDoRA Grid Search (CIFAR-100, EMA 0.3 @ Round {EMA_TARGET_ROUND}) — {len(done)}/{len(rows)} parsed")
    print("=" * 78)
    for part in PARTITIONS:
        pr = sorted([r for r in done if r["part"] == part], key=lambda r: r["ema"], reverse=True)
        if not pr:
            continue
        print(f"\n[{part}]  ({len(pr)} runs)")
        for r in pr[:8]:
            best = "  <= BEST" if r is pr[0] else ""
            print(f"   EMA={r['ema']:.4f}  rounds={r['n']:>2}  {r['cfg']}{best}")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.summarize:
        summarize(); return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tasks = enumerate_tasks(script_dir)
    print(f"[{time.strftime('%H:%M:%S')}] FeDoRA CIFAR-100 grid: {len(tasks)} runs "
          f"(lr×{len(GRID['lr'])}, alpha×{len(GRID['lora_alpha'])}, dora_m_lr×{len(GRID['dora_m_lr'])} "
          f"× {len(PARTITIONS)} partitions), {ROUNDS} rounds, GPUs={args.gpus}", flush=True)
    print(f"  python={sys.executable}", flush=True)

    if args.dry_run:
        for t in tasks[:6]:
            print("   " + t[0])
        print(f"   ... (+{len(tasks)-6} more)")
        return

    # 완료(>=ROUNDS) 건너뛰기 → 중단/재연결 후 이어하기
    task_queue = queue.Queue(); skipped = 0
    for tag, cmd in tasks:
        lp = os.path.join(LOG_DIR, f"{tag}.log")
        if os.path.exists(lp) and len(get_accs(lp)) >= EMA_TARGET_ROUND:
            skipped += 1; continue
        task_queue.put((tag, cmd))
    print(f"  queued {task_queue.qsize()} (skipped {skipped} done)", flush=True)

    threads = []
    for g in args.gpus:
        t = threading.Thread(target=worker, args=(g, task_queue)); t.start(); threads.append(t)
    for t in threads:
        t.join()
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE. Run --summarize to rank.", flush=True)


if __name__ == "__main__":
    main()
