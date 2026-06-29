"""
FeDoRA Hyperparameter Grid Search (FeDoRA 전용, 기법 불변)
=========================================================
제안 기법 FeDoRA가 가장 좋은 성능을 내는 하이퍼파라미터를 탐색한다.

★ 기법은 변경하지 않는다 ★
  고정 인자: --peft dora --flex_lora --flex_lora_svd_a --target_modules qkv
  (DoRA 분해 + full-weight FedAvg + client-A SVD 기반 SVD-A 집계, warm-up 미사용)
  탐색 대상은 '하이퍼파라미터 값'일 뿐 알고리즘 구조가 아니다.

────────────────────────────────────────────────────────────────────────
그리드서치 일반 상식 (Q: HP 1개당 후보 몇 개?)
  - 보통 HP 1개당 후보 3~5개.
      · 학습률(lr): 로그 스케일로 3~4개 (예: 5e-4, 1e-3, 5e-3, 1e-2)
      · 그 외:      2~4개
  - 핵심 HP 2~3개를 동시에 격자 탐색 → 환경당 보통 9~36 조합('평균' 규모).
  - HP 수를 늘리면 조합이 곱으로 폭증(예: 4개 HP × 각 3값 = 81)하므로,
    실무에선 핵심 HP 소수만 grid, 나머지는 고정/coarse-to-fine.
────────────────────────────────────────────────────────────────────────

탐색 가능한 FeDoRA 하이퍼파라미터 (기법 불변):
  --lr          A/B/classifier 학습률         ★ 가장 중요
  --lora_alpha  LoRA scaling (alpha/r)         ★ 중요 (업데이트 크기)
  --dora_m_lr   DoRA magnitude(m) 전용 학습률   ★ DoRA 특유 노브
  --lora_r      rank (용량/통신량; 보통 32 고정)
  --reg         weight decay (A/B L2)
  --local_steps 라운드당 local update step 수
  --eta_min     cosine scheduler 최저 학습률

프리셋 (--preset):
  quick   : lr(3) × lora_alpha(2)                =  6 조합/환경   (빠른 감잡기)
  medium  : lr(4) × lora_alpha(3) × dora_m_lr(3) = 36 조합/환경   (평균/권장)
  full    : medium + reg(2) + local_steps(2)     = 144 조합/환경  (철저, 매우 큼)

사용법:
  python3 run_fedora_gridsearch.py --preset medium --dry-run                 # 계획/조합수 확인
  python3 run_fedora_gridsearch.py --preset medium --datasets cifar100 --partitions noniid
  python3 run_fedora_gridsearch.py --preset quick                            # 전체 환경 빠른 탐색
  python3 run_fedora_gridsearch.py --summarize                               # 환경별 best 요약

로그: logs/fedora_gridsearch/{dataset}/{run_tag}.log  (메인 실험 비파괴, 별도 디렉토리)
"""

import os
import re
import time
import queue
import argparse
import itertools
import threading
import subprocess

# =====================================================================
# 고정값 — 기법(FeDoRA) 정의. 변경 금지.
# =====================================================================
FEDORA_FIXED_ARGS = "--peft dora --flex_lora --flex_lora_svd_a --target_modules qkv"
MODEL = "vit-base"
ROUNDS = 50
EPOCHS = 1
SEED = 42
DATADIR = "/mnt/data1"
BETA = 0.3                            # Dirichlet alpha (Non-IID)
SAMPLE_FRACTION = {20: 0.15, 50: 0.06}   # 3/|C|
EMA_ALPHA = 0.7                      # 결과표와 동일한 EMA(0.3): e = 0.7e + 0.3x
EMA_TARGET_ROUND = 50
LOG_BASE = "./logs/fedora_gridsearch"

# 환경 축 (기본 전체)
DEFAULT_ENVS = {"datasets": ["cifar100", "svhn"], "clients": [20, 50], "partitions": ["iid", "noniid"]}

# =====================================================================
# 하이퍼파라미터 후보군 — 프리셋별. 'same'=--dora_m_lr 미전달(lr 공유), 'none'=미전달.
# =====================================================================
PRESETS = {
    "quick": {
        "lr":          ["1e-3", "5e-3", "1e-2"],
        "lora_alpha":  [16, 32],
        "lora_r":      [32],
        "dora_m_lr":   ["same"],
        "reg":         ["0"],
        "local_steps": [50],
        "eta_min":     ["0"],
    },
    "medium": {   # ← 권장(평균 규모): 핵심 HP 3개
        "lr":          ["5e-4", "1e-3", "5e-3", "1e-2"],
        "lora_alpha":  [16, 32, 64],
        "lora_r":      [32],
        "dora_m_lr":   ["same", "1e-3", "5e-4"],
        "reg":         ["0"],
        "local_steps": [50],
        "eta_min":     ["0"],
    },
    "full": {     # 철저(매우 큼): + reg, local_steps
        "lr":          ["5e-4", "1e-3", "5e-3", "1e-2"],
        "lora_alpha":  [16, 32, 64],
        "lora_r":      [32],
        "dora_m_lr":   ["same", "1e-3", "5e-4"],
        "reg":         ["0", "1e-4"],
        "local_steps": [50, 100],
        "eta_min":     ["0"],
    },
}
HP_KEYS = ["lr", "lora_alpha", "lora_r", "dora_m_lr", "reg", "local_steps", "eta_min"]


def build_run_tag(env, hp):
    def s(x):
        return str(x).replace(".", "p").replace("-", "m")
    return (f"fedora_{env['dataset']}_{env['clients']}c_{env['partition']}"
            f"_lr{s(hp['lr'])}_a{hp['lora_alpha']}_r{hp['lora_r']}_mlr{s(hp['dora_m_lr'])}"
            f"_reg{s(hp['reg'])}_ls{hp['local_steps']}_em{s(hp['eta_min'])}_seed{SEED}")


def build_command(env, hp, script_dir, log_dir, run_tag):
    sf = SAMPLE_FRACTION[env["clients"]]
    common = (
        f"--model {MODEL} --dataset {env['dataset']} --datadir {DATADIR} --round {ROUNDS} "
        f"--epochs {EPOCHS} --n_clients {env['clients']} --sample_fraction {sf} "
        f"--seed {SEED} --optimizer adam --lr {hp['lr']} --reg {hp['reg']} "
        f"--scheduler cosine --eta_min {hp['eta_min']} "
        f"--local_steps {hp['local_steps']} --batch_size 32 "
    )
    common += (f"--partition noniid --beta {BETA} " if env["partition"] == "noniid" else "--partition iid ")
    tune = f"--lora_r {hp['lora_r']} --lora_alpha {hp['lora_alpha']} "
    if hp["dora_m_lr"] != "same":
        tune += f"--dora_m_lr {hp['dora_m_lr']} "
    return (f"cd {script_dir} && python3 main.py "
            f"{common} {FEDORA_FIXED_ARGS} {tune} --ft_classifier "
            f"--logdir {log_dir} --log_file_name '{run_tag}'")


def enumerate_tasks(envs, grid, script_dir):
    tasks = []
    for dataset in envs["datasets"]:
        for clients in envs["clients"]:
            for partition in envs["partitions"]:
                env = {"dataset": dataset, "clients": clients, "partition": partition}
                log_dir = os.path.join(LOG_BASE, dataset)
                for combo in itertools.product(*[grid[k] for k in HP_KEYS]):
                    hp = dict(zip(HP_KEYS, combo))
                    tag = build_run_tag(env, hp)
                    tasks.append((tag, build_command(env, hp, script_dir, log_dir, tag), log_dir))
    return tasks


# ----------------------- GPU 워커 (VRAM 관리) -----------------------
def wait_for_gpu_free_by_vram(gpu_id, vram_threshold_mb=5000, check_interval=30):
    first = True
    while True:
        try:
            used = int(subprocess.check_output(
                f"nvidia-smi -i {gpu_id} --query-gpu=memory.used --format=csv,noheader,nounits",
                shell=True, text=True).strip())
            if used < vram_threshold_mb:
                if not first:
                    print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} free ({used} MB).")
                break
            if first:
                print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} BUSY ({used} MB). Waiting...")
                first = False
            time.sleep(check_interval)
        except subprocess.CalledProcessError:
            time.sleep(check_interval)


def run_command(command, gpu_id, task_name, log_dir):
    print(f"[{time.strftime('%H:%M:%S')}] Start '{task_name}' GPU{gpu_id}")
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu_id); env["PYTHONUNBUFFERED"] = "1"
    lp = os.path.join(log_dir, f"{task_name}.log"); os.makedirs(os.path.dirname(lp), exist_ok=True)
    with open(lp, "w") as f:
        p = subprocess.Popen(command, shell=True, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line); f.flush()
        p.wait()
    print(f"[{time.strftime('%H:%M:%S')}] Done '{task_name}' [{'OK' if p.returncode==0 else 'FAIL'}]")


def worker(gpu_id, task_queue):
    while True:
        try:
            tn, cmd, ld = task_queue.get_nowait()
        except queue.Empty:
            break
        wait_for_gpu_free_by_vram(gpu_id)
        run_command(cmd, gpu_id, tn, ld)
        task_queue.task_done()


# ----------------------- 결과 파싱 / 요약 -----------------------
def get_accs(p):
    accs, seen = [], set()
    pat = re.compile(r'(\d{2}-\d{2} \d{2}:\d{2}).*>> Global Model Test accuracy: ([0-9.]+)')
    for line in open(p, errors='ignore'):
        m = pat.search(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1)); accs.append(float(m.group(2)))
    return accs


def compute_ema(accs, alpha=EMA_ALPHA, target=EMA_TARGET_ROUND):
    if not accs:
        return None
    e = accs[0]; vals = [e]
    for v in accs[1:]:
        e = alpha * e + (1 - alpha) * v; vals.append(e)
    return vals[min(target - 1, len(vals) - 1)]


def summarize():
    if not os.path.isdir(LOG_BASE):
        print(f"No logs at {LOG_BASE}"); return
    tag_pat = re.compile(r"fedora_(?P<ds>[a-z0-9]+)_(?P<nc>\d+)c_(?P<part>iid|noniid)_(?P<rest>.+)")
    rows = []
    for ds in os.listdir(LOG_BASE):
        d = os.path.join(LOG_BASE, ds)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".log"):
                continue
            m = tag_pat.match(fn[:-4])
            if not m:
                continue
            accs = get_accs(os.path.join(d, fn))
            rows.append({"env": f"{m['ds']}_{m['nc']}c_{m['part']}", "config": m["rest"],
                         "rounds": len(accs), "ema": compute_ema(accs)})
    done = [r for r in rows if r["ema"] is not None]
    if not done:
        print("No completed runs yet."); return
    print("=" * 80)
    print("FeDoRA Grid Search — Best config per environment (EMA 0.3 @ Round 50)")
    print("=" * 80)
    for env in sorted(set(r["env"] for r in done)):
        er = sorted([r for r in done if r["env"] == env], key=lambda r: r["ema"], reverse=True)
        print(f"\n[{env}]  ({len(er)} runs)")
        for r in er[:5]:
            best = "  <= BEST" if r is er[0] else ""
            print(f"   EMA={r['ema']:.4f}  rounds={r['rounds']:>2}  {r['config']}{best}")


# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser(description="FeDoRA-only hyperparameter grid search")
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="medium")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_ENVS["datasets"])
    ap.add_argument("--clients", type=int, nargs="+", default=DEFAULT_ENVS["clients"])
    ap.add_argument("--partitions", nargs="+", default=DEFAULT_ENVS["partitions"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.summarize:
        summarize(); return

    grid = PRESETS[args.preset]
    envs = {"datasets": args.datasets, "clients": args.clients, "partitions": args.partitions}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tasks = enumerate_tasks(envs, grid, script_dir)

    n_env = len(args.datasets) * len(args.clients) * len(args.partitions)
    n_hp = 1
    for k in HP_KEYS:
        n_hp *= len(grid[k])
    print(f"[preset={args.preset}] HParam combos/env = {n_hp}  ×  environments = {n_env}  =  {len(tasks)} runs")
    print(f"  per-HP 후보: " + ", ".join(f"{k}={len(grid[k])}" for k in HP_KEYS if len(grid[k]) > 1))
    print(f"  (run당 ViT 50R ≈ 1.5~6h, GPU {len(args.gpus)} 병렬)")

    if args.dry_run:
        for t in tasks[:8]:
            print("   " + t[0])
        if len(tasks) > 8:
            print(f"   ... (+{len(tasks)-8} more)")
        return

    # 완료(>=50R) 건너뛰기 → 중단 후 이어하기
    task_queue = queue.Queue(); skipped = 0
    for tn, cmd, ld in tasks:
        lp = os.path.join(ld, f"{tn}.log")
        if os.path.exists(lp) and len(get_accs(lp)) >= EMA_TARGET_ROUND:
            skipped += 1; continue
        task_queue.put((tn, cmd, ld))
    print(f"Queued {task_queue.qsize()} (skipped {skipped} done). Starting...")
    print("=" * 60)

    threads = []
    for g in args.gpus:
        t = threading.Thread(target=worker, args=(g, task_queue)); t.start(); threads.append(t)
    for t in threads:
        t.join()
    print(f"[{time.strftime('%H:%M:%S')}] Grid search done. Run --summarize to rank.")


if __name__ == "__main__":
    main()
