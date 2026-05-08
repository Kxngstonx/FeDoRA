import os
import subprocess
import time
import queue
import threading
import argparse

BETAS = [0.5, 0.3]
GLUE_TASKS = ['sst2', 'mnli', 'qnli', 'qqp']

FL_BASE = (
    "--round 50 --epochs 1 "
    "--n_clients 100 --sample_fraction 0.1 "
    "--seed 42 "
    "--optimizer adam --lr 1e-4 --scheduler cosine "
    "--partition noniid "
    "--lora_r 8 --lora_alpha 16"
)

METHODS = {
    'LoRA':           '--peft lora',
    'LoRA_TrainableA':'--peft lora --trainable_A',
    'FlexLoRA':       '--peft dora --flex_lora',
}


def run_command(command, gpu_id, task_name, log_dir):
    print(f"[{time.strftime('%H:%M:%S')}] Starting {task_name} on GPU {gpu_id}...\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    log_file_path = os.path.join(log_dir, f"{task_name.lower()}_gpu{gpu_id}.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    with open(log_file_path, "w") as f:
        process = subprocess.Popen(
            command,
            shell=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            f.write(line)
            f.flush()
        process.wait()

    print(f"\n[{time.strftime('%H:%M:%S')}] Finished {task_name} on GPU {gpu_id}.")


def worker(gpu_id, task_queue, dry_run):
    while not task_queue.empty():
        try:
            task_name, command, log_dir = task_queue.get(timeout=1)
            if dry_run:
                print(f"[DRY-RUN] {task_name}\n  {command}\n")
            else:
                run_command(command, gpu_id, task_name, log_dir)
            task_queue.task_done()
        except queue.Empty:
            break


def build_tasks():
    tasks = []

    # --- Group 1: RoBERTa-Large + AG News ---
    for beta in BETAS:
        log_dir = f'./logs/roberta_agnews/beta_{beta}/'
        base = f'python3 main.py --model roberta-large --dataset ag_news {FL_BASE} --beta {beta} --logdir {log_dir}'
        for method_name, method_flags in METHODS.items():
            tag = f'RoBERTa_AGNews_Beta{beta}_{method_name}'
            cmd = f'{base} {method_flags} --log_file_name {tag}'
            tasks.append((tag, cmd, log_dir))

    # --- Group 2: RoBERTa-Large + GLUE (per task) ---
    for task in GLUE_TASKS:
        for beta in BETAS:
            log_dir = f'./logs/roberta_glue/{task}/beta_{beta}/'
            base = f'python3 main.py --model roberta-large --dataset glue_{task} {FL_BASE} --beta {beta} --logdir {log_dir}'
            for method_name, method_flags in METHODS.items():
                tag = f'RoBERTa_{task.upper()}_Beta{beta}_{method_name}'
                cmd = f'{base} {method_flags} --log_file_name {tag}'
                tasks.append((tag, cmd, log_dir))

    # --- Group 3: Qwen + GLUE (per task) ---
    for task in GLUE_TASKS:
        for beta in BETAS:
            log_dir = f'./logs/qwen_glue/{task}/beta_{beta}/'
            base = f'python3 main.py --model qwen --dataset glue_{task} {FL_BASE} --beta {beta} --logdir {log_dir}'
            for method_name, method_flags in METHODS.items():
                tag = f'Qwen_{task.upper()}_Beta{beta}_{method_name}'
                cmd = f'{base} {method_flags} --log_file_name {tag}'
                tasks.append((tag, cmd, log_dir))

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1], help='GPU IDs to use')
    args = parser.parse_args()

    tasks = build_tasks()

    task_queue = queue.Queue()
    for t in tasks:
        task_queue.put(t)

    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks)} tasks across {len(args.gpus)} GPU(s): {args.gpus}\n")

    threads = []
    for gpu_id in args.gpus:
        t = threading.Thread(target=worker, args=(gpu_id, task_queue, args.dry_run))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"[{time.strftime('%H:%M:%S')}] All experiments completed.")


if __name__ == "__main__":
    main()
