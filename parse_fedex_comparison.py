"""
FedEx 시나리오 비교표 추출
==========================
원본 FedEx-LoRA(메인 로그) + W0-frozen/faircomm 시나리오 로그 + 기준선(FeDoRA, FedIT)을
EMA(0.3)@Round50 으로 환산하여 환경별 비교표(markdown)를 출력/저장한다.

  - 메인 vision 로그:      logs/vit_cifar100_svhn_experiments/{ds}_lower/
  - FedEx 시나리오 로그:   logs/vit_fedex_scenarios/{ds}_lower/

사용법:
  python3 parse_fedex_comparison.py            # 콘솔 출력
  python3 parse_fedex_comparison.py --md out.md  # markdown 파일로 저장
"""
import re
import os
import argparse

EMA_ALPHA = 0.7
EMA_ROUND = 50
MAIN = "logs/vit_cifar100_svhn_experiments"
SCEN = "logs/vit_fedex_scenarios"


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


def ema(accs, al=EMA_ALPHA, t=EMA_ROUND):
    if not accs:
        return None
    e = accs[0]; vals = [e]
    for v in accs[1:]:
        e = al * e + (1 - al) * v; vals.append(e)
    return vals[min(t - 1, len(vals) - 1)]


def val(path):
    a = get_accs(path)
    v = ema(a)
    return (f"{v:.4f}" if v is not None else "  -  "), len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None, help="markdown 저장 경로")
    args = ap.parse_args()

    # (라벨, 로그 경로 생성 함수)
    def main_log(ds, nc, part, method):
        return os.path.join(MAIN, f"{ds}_lower", f"{ds}_{nc}c_{part}_lower_{method}_seed42.log")

    def scen_log(ds, nc, part, scen):
        return os.path.join(SCEN, f"{ds}_lower", f"{ds}_{nc}c_{part}_lower_FedEx-LoRA-{scen}_seed42.log")

    rows_def = [
        ("FeDoRA (ours, frozen W0)",        lambda ds, nc, p: main_log(ds, nc, p, "FeDoRA")),
        ("FedIT (frozen W0)",               lambda ds, nc, p: main_log(ds, nc, p, "FedIT")),
        ("FedEx-LoRA (orig, dense W0 ~19x)",lambda ds, nc, p: main_log(ds, nc, p, "FedEx-LoRA")),
        ("FedEx-LoRA-fairW0 (=FedIT, 1x)",  lambda ds, nc, p: scen_log(ds, nc, p, "fairW0")),
        ("FedEx-LoRA-faircomm16 (DesignA,1x)", lambda ds, nc, p: scen_log(ds, nc, p, "faircomm16")),
        ("FedEx-LoRA-faircomm32 (DesignA,2x)", lambda ds, nc, p: scen_log(ds, nc, p, "faircomm32")),
    ]

    out = []
    out.append("# FedEx Scenario Comparison (EMA 0.3 @ Round 50)\n")
    out.append("통신 비용 주석: orig≈19×, faircomm16≈1×(동일예산), faircomm32≈2×, fairW0=1×(=FedIT)\n")
    for nc in [20, 50]:
        out.append(f"\n## {nc} Clients\n")
        header = "| Method | C100 IID | C100 NonIID | SVHN IID | SVHN NonIID |"
        out.append(header)
        out.append("|" + "---|" * 5)
        for label, fn in rows_def:
            cells = []
            for ds, part in [("cifar100", "iid"), ("cifar100", "noniid"),
                             ("svhn", "iid"), ("svhn", "noniid")]:
                v, n = val(fn(ds, nc, part))
                mark = "" if n >= EMA_ROUND or v.strip() == "-" else f"({n})"
                cells.append(f"{v}{mark}")
            out.append(f"| {label} | " + " | ".join(cells) + " |")

    text = "\n".join(out)
    print(text)
    if args.md:
        with open(args.md, "w") as f:
            f.write(text + "\n")
        print(f"\n[saved] {args.md}")


if __name__ == "__main__":
    main()
