"""
Phase 오케스트레이터 — 전 기법 30라운드 일관 비교 (ViT-B/16, seed 42)
=====================================================================
★ 모든 비교 기법을 동일하게 30 rounds 로 새로 실행한다.
  (기존 50라운드 로그에서 @30 추출은 cosine LR 스케줄이 라운드 수에 의존하므로 불공정 → 재실행)
★ FedEx-LoRA 는 전 환경에서 '수정 버전(Design A, faircomm 저랭크 잔차 통신, 동일 예산
  lora_r16+res16=32)' 사용.

Phase 1 (SVHN, 20 clients) = 82 runs
  - FeDoRA 그리드서치: lr×4 {5e-4,1e-3,5e-3,1e-2} · lora_alpha×3 {16,32,64}
                       · dora_m_lr×3 {same,1e-3,1e-4} × {IID,NonIID} = 72
  - 비교 기법 × {IID,NonIID}:
       FedEx-LoRA-mod, FedIT, FlexLoRA, FFA-LoRA, RAVAN = 5 × 2 = 10

Phase 2 (CIFAR-100 & SVHN, 50 clients) — Phase1 종료 후 = 22 runs
  - FeDoRA(이전 그리드 best HP): CIFAR=완료된 20c 그리드 best,
                                 SVHN=Phase1 그리드 best(런타임 파싱) → 2ds×2part = 4
  - FedEx-LoRA-mod, FedIT, FlexLoRA, FFA-LoRA × 2ds × 2part = 16
  - RAVAN: SVHN 만 (CIFAR-100 은 init 단계 실패) × 2part = 2

총 104 runs. 완료(>=30R) 자동 skip(재연결/재실행 안전).

로그:
  FeDoRA grid       : logs/fedora_gridsearch_svhn30/svhn/
  비교(20c·50c 전부): logs/comparison_30r/{ds}/{ds}_{nc}c_{part}_lower_{Method}_seed42.log
"""
import os
import re
import sys
import time
import queue
import itertools
import threading
import subprocess

PY = sys.executable
SD = os.path.dirname(os.path.abspath(__file__))
DATADIR = "/mnt/data1"; MODEL = "vit-base"; ROUNDS = 30; SEED = 42; BETA = 0.3
LOCAL_STEPS = 50; BATCH = 32; SF = {20: 0.15, 50: 0.06}; GPUS = [0, 1]

FEDORA_FIXED = "--peft dora --flex_lora --flex_lora_svd_a --target_modules qkv"
GRID = {"lr": ["5e-4", "1e-3", "5e-3", "1e-2"], "lora_alpha": [16, 32, 64],
        "dora_m_lr": ["same", "1e-3", "1e-4"]}
# ★ CIFAR-100 전용 그리드 best HP. 다른 데이터셋(SVHN 등)에는 절대 사용하지 않는다.
#   HP는 태스크/데이터셋마다 최적값이 다르므로, 각 데이터셋은 '자기 자신의' 그리드 결과만 쓴다.
CIFAR100_GRID_BEST = {"iid": ("1e-3", 16, "1e-4"), "noniid": ("1e-3", 32, "1e-3")}

# 비교 기법: name -> (args, lr per dataset, datasets-to-run)
FEDEX_MOD_ARGS = ("--peft lora --fedex_lora --fedex_lowrank_comm --fedex_res_rank 16 "
                  "--trainable_A --lora_r 16 --lora_alpha 16 --target_modules qkv")
# 베이스라인(FedIT/FlexLoRA/FFA-LoRA/RAVAN)은 기존 메인 로그에서 EMA@30/Max 추출 → 재실행 안 함.
# 신규 실행 대상은 'FedEx 수정버전(Design A)' 만 (기존 로그엔 원본 FedEx 뿐이라 새로 필요).
METHODS = {
    "FedEx-LoRA-mod": (FEDEX_MOD_ARGS,
                       {"cifar100": "1e-3", "svhn": "1e-3"}, ["cifar100", "svhn"]),
}
GRID_DIR = "./logs/fedora_gridsearch_svhn30/svhn"
CMP_DIR = "./logs/comparison_30r/{ds}"


def st(x): return str(x).replace(".", "p").replace("-", "m")


def common(ds, nc, part, lr):
    a = (f"--model {MODEL} --dataset {ds} --datadir {DATADIR} --round {ROUNDS} --epochs 1 "
         f"--n_clients {nc} --sample_fraction {SF[nc]} --seed {SEED} --optimizer adam --lr {lr} "
         f"--reg 0 --scheduler cosine --local_steps {LOCAL_STEPS} --batch_size {BATCH} ")
    return a + (f"--partition noniid --beta {BETA} " if part == "noniid" else "--partition iid ")


def fedora_cmd(ds, nc, part, lr, a, m, logdir, tag):
    tune = f"--lora_r 32 --lora_alpha {a} " + (f"--dora_m_lr {m} " if m != "same" else "")
    return (f"cd {SD} && {PY} main.py {common(ds,nc,part,lr)} {FEDORA_FIXED} {tune}"
            f"--ft_classifier --logdir {logdir} --log_file_name '{tag}'")


def method_cmd(name, ds, nc, part, logdir, tag):
    args, lrmap, _ = METHODS[name]
    return (f"cd {SD} && {PY} main.py {common(ds,nc,part,lrmap[ds])} {args} "
            f"--ft_classifier --logdir {logdir} --log_file_name '{tag}'")


# ---------- GPU 워커 ----------
def gpu_free(g, thr=5000, iv=30):
    f = True
    while True:
        try:
            u = int(subprocess.check_output(
                f"nvidia-smi -i {g} --query-gpu=memory.used --format=csv,noheader,nounits",
                shell=True, text=True).strip())
            if u < thr:
                return
            if f:
                print(f"[{time.strftime('%H:%M:%S')}] GPU{g} busy({u}MB) wait", flush=True); f = False
            time.sleep(iv)
        except subprocess.CalledProcessError:
            time.sleep(iv)


def get_accs(p):
    a, s = [], set(); pat = re.compile(r'(\d{2}-\d{2} \d{2}:\d{2}).*Test accuracy: ([0-9.]+)')
    if not os.path.exists(p): return a
    for l in open(p, errors='ignore'):
        m = pat.search(l)
        if m and m.group(1) not in s: s.add(m.group(1)); a.append(float(m.group(2)))
    return a


def run_one(cmd, g, tag, ld):
    print(f"[{time.strftime('%H:%M:%S')}] START {tag} GPU{g}", flush=True)
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(g); env["PYTHONUNBUFFERED"] = "1"
    os.makedirs(ld, exist_ok=True)
    with open(os.path.join(ld, f"{tag}.log"), "w") as f:
        p = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line); f.flush()
        p.wait()
    print(f"[{time.strftime('%H:%M:%S')}] DONE {tag} [{'OK' if p.returncode==0 else 'FAIL'}]", flush=True)


def run_queue(tasks):
    q = queue.Queue(); skip = 0
    for tag, cmd, ld in tasks:
        if len(get_accs(os.path.join(ld, f"{tag}.log"))) >= ROUNDS: skip += 1; continue
        q.put((tag, cmd, ld))
    print(f"[{time.strftime('%H:%M:%S')}] queue {q.qsize()} (skip {skip})", flush=True)

    def wk(g):
        while True:
            try: tag, cmd, ld = q.get_nowait()
            except queue.Empty: return
            gpu_free(g); run_one(cmd, g, tag, ld); q.task_done()
    ts = [threading.Thread(target=wk, args=(g,)) for g in GPUS]
    for t in ts: t.start()
    for t in ts: t.join()


# ---------- Phase 1 ----------
def phase1():
    print("="*60, flush=True); print("PHASE 1: SVHN 20c — FeDoRA grid(72) + FedEx-mod(2) = 74 (베이스라인은 기존로그)", flush=True); print("="*60, flush=True)
    tasks = []
    for part in ["iid", "noniid"]:
        for lr, a, m in itertools.product(GRID["lr"], GRID["lora_alpha"], GRID["dora_m_lr"]):
            tag = f"fedora_svhn_20c_{part}_lr{st(lr)}_a{a}_mlr{st(m)}_r32_R{ROUNDS}_seed{SEED}"
            tasks.append((tag, fedora_cmd("svhn", 20, part, lr, a, m, GRID_DIR, tag), GRID_DIR))
        cd = CMP_DIR.format(ds="svhn")
        for name in METHODS:
            if "svhn" not in METHODS[name][2]: continue
            tag = f"svhn_20c_{part}_lower_{name}_seed{SEED}"
            tasks.append((tag, method_cmd(name, "svhn", 20, part, cd, tag), cd))
    run_queue(tasks)


def find_svhn_best():
    def ema(a, al=0.7, t=ROUNDS):
        if not a: return None
        e = a[0]; v = [e]
        for x in a[1:]: e = al*e+(1-al)*x; v.append(e)
        return v[min(t-1, len(v)-1)]
    pat = re.compile(r"fedora_svhn_20c_(iid|noniid)_lr(\w+?)_a(\d+)_mlr(\w+?)_r32")
    best = {}
    for fn in (os.listdir(GRID_DIR) if os.path.isdir(GRID_DIR) else []):
        if not fn.endswith(".log"): continue
        m = pat.match(fn)
        if not m: continue
        e = ema(get_accs(os.path.join(GRID_DIR, fn)))
        if e is None: continue
        p = m.group(1); lr = m.group(2).replace("p", ".").replace("m", "-"); a = int(m.group(3))
        ml = "same" if m.group(4) == "same" else m.group(4).replace("p", ".").replace("m", "-")
        if p not in best or e > best[p][0]: best[p] = (e, lr, a, ml)
    return best


# ---------- Phase 2 ----------
def phase2(svb):
    print("="*60, flush=True); print("PHASE 2: 50c CIFAR-100 & SVHN — FeDoRA(grid) + FedEx-mod = 8", flush=True); print("="*60, flush=True)
    # ★ 각 데이터셋은 '자기 자신의' 그리드 best HP만 사용한다.
    #   CIFAR-100 → CIFAR-100 그리드 best, SVHN → SVHN 그리드 best.
    #   SVHN 그리드 best가 없으면(불완전) SVHN FeDoRA-grid 실행을 '건너뛴다'.
    #   어떤 경우에도 CIFAR HP를 SVHN(또는 그 반대)에 대체 사용하지 않는다.
    HP = {"cifar100": CIFAR100_GRID_BEST}
    if len(svb) == 2:
        HP["svhn"] = {p: (svb[p][1], svb[p][2], svb[p][3]) for p in svb}
    else:
        print(f"WARN: SVHN grid best 불완전({list(svb)}) → SVHN 50c FeDoRA-grid 건너뜀 "
              f"(CIFAR HP 대체 사용 금지)", flush=True)
    print(f"Phase2 FeDoRA HP — CIFAR-100:{CIFAR100_GRID_BEST} | SVHN:{HP.get('svhn','SKIP')}", flush=True)
    tasks = []
    for ds in ["cifar100", "svhn"]:
        cd = CMP_DIR.format(ds=ds)
        for part in ["iid", "noniid"]:
            # FeDoRA-grid: 해당 데이터셋의 grid best HP가 존재할 때만 실행
            if ds in HP and part in HP[ds]:
                lr, a, m = HP[ds][part]
                tag = f"{ds}_50c_{part}_lower_FeDoRA-grid_seed{SEED}"
                tasks.append((tag, fedora_cmd(ds, 50, part, lr, a, m, cd, tag), cd))
            # 비교 기법(FedEx-mod 등)은 grid HP와 무관하므로 그대로 실행
            for name in METHODS:
                if ds not in METHODS[name][2]: continue
                t2 = f"{ds}_50c_{part}_lower_{name}_seed{SEED}"
                tasks.append((t2, method_cmd(name, ds, 50, part, cd, t2), cd))
    run_queue(tasks)


def main():
    t0 = time.time()
    phase1()
    sb = find_svhn_best()
    print(f"[{time.strftime('%H:%M:%S')}] SVHN 20c grid best: {sb}", flush=True)
    phase2(sb)
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE ({(time.time()-t0)/3600:.1f}h)", flush=True)


if __name__ == "__main__":
    main()
