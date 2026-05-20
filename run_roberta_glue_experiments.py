import os
import subprocess
import time
import queue
import threading
import argparse

BETAS = [0.5]
GLUE_TASKS = ['sst2', 'mnli', 'qnli', 'qqp']

FL_BASE = (
    "--round 30 --epochs 1 "
    "--n_clients 3 --sample_fraction 1.0 "
    "--seed 42 "
    "--optimizer sgd --lr 1e-4 --scheduler cosine "
    "--partition noniid "
    "--lora_r 8 --lora_alpha 8 "
    "--local_steps 50 "
    "--ft_classifier"
)

METHODS = {
    'LoRA_TrainableA':  '--peft lora --trainable_A',
    'FlexLoRA':         '--peft dora --flex_lora',
    'FlexLoRA_FreezeA': '--peft dora --flex_lora --flex_lora_freeze_a',
}


def run_command(command, gpu_id, task_name, log_dir):
    print(f"[{time.strftime('%H:%M:%S')}] Starting {task_name} on GPU {gpu_id}...\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, task_name + '.log')

    with open(log_path, 'a') as log_f:
        process = subprocess.Popen(
            command,
            shell=True,
            env=env,
            stdout=log_f,
            stderr=log_f,
        )
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

    for task in GLUE_TASKS:
        for beta in BETAS:
            log_dir = f'./logs/roberta_glue/{task}/'
            base = f'python3 main.py --model roberta-large --dataset glue_{task} {FL_BASE} --beta {beta} --logdir {log_dir}'
            for method_name, method_flags in METHODS.items():
                tag = f'RoBERTa_{task.upper()}_Beta{beta}_{method_name}'
                cmd = f'{base} {method_flags} --log_file_name {tag}'
                tasks.append((tag, cmd, log_dir))

    return tasks


def get_free_gpu_ids(threshold_mib=40000):
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    free = []
    for line in result.stdout.strip().split('\n'):
        idx, mem = line.split(', ')
        if int(mem) >= threshold_mib:
            free.append(int(idx))
    return free


def gpu_monitor(task_queue, active_gpus, threads, dry_run, poll_interval=60):
    while not task_queue.empty():
        for gpu_id in get_free_gpu_ids():
            if gpu_id not in active_gpus and not task_queue.empty():
                print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} is now free — spawning worker.")
                active_gpus.add(gpu_id)
                t = threading.Thread(target=worker, args=(gpu_id, task_queue, dry_run))
                t.start()
                threads.append(t)
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--gpus', type=int, nargs='+', default=None, help='GPU IDs to use (default: auto-detect free GPUs)')
    parser.add_argument('--mem-threshold', type=int, default=40000, help='Free memory threshold in MiB to consider a GPU available')
    args = parser.parse_args()

    tasks = build_tasks()

    task_queue = queue.Queue()
    for t in tasks:
        task_queue.put(t)

    if args.gpus is not None:
        initial_gpus = args.gpus
    else:
        initial_gpus = get_free_gpu_ids(args.mem_threshold)
        if not initial_gpus:
            print(f"[{time.strftime('%H:%M:%S')}] No free GPUs found (threshold: {args.mem_threshold} MiB). Waiting...")
            while not initial_gpus:
                time.sleep(30)
                initial_gpus = get_free_gpu_ids(args.mem_threshold)

    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks)} tasks, starting on GPU(s): {initial_gpus}\n")

    active_gpus = set(initial_gpus)
    threads = []
    for gpu_id in initial_gpus:
        t = threading.Thread(target=worker, args=(gpu_id, task_queue, args.dry_run))
        t.start()
        threads.append(t)

    monitor = threading.Thread(
        target=gpu_monitor,
        args=(task_queue, active_gpus, threads, args.dry_run),
        daemon=True
    )
    monitor.start()

    for t in threads:
        t.join()

    print(f"[{time.strftime('%H:%M:%S')}] All experiments completed.")


if __name__ == "__main__":
    main()
