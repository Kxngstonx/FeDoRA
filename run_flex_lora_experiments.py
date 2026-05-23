import os
import subprocess
import time
import queue
import threading
import json
import pickle
import torch


def export_lambda_logs(pkl_path, json_path):
    if not os.path.exists(pkl_path):
        print(f"  [warn] pkl not found: {pkl_path}")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    lambda_data = data.get('flex_dora_sv', [])
    if not lambda_data:
        print(f"  [warn] flex_dora_sv is empty in {pkl_path}")
        return

    serializable = []
    for round_logs in lambda_data:
        round_dict = {}
        for layer, sv in round_logs.items():
            round_dict[layer] = sv.tolist() if isinstance(sv, torch.Tensor) else sv
        serializable.append(round_dict)

    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"  [info] lambda_logs saved → {json_path} ({len(serializable)} rounds)")


def run_command(command, gpu_id, task_name, log_dir, log_file_name):
    print(f"[{time.strftime('%H:%M:%S')}] Starting {task_name} on GPU {gpu_id}...\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file_name + '.log')

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

    pkl_path = os.path.join(log_dir, f"{log_file_name}.pkl")
    json_path = os.path.join(log_dir, f"{log_file_name}_lambda.json")
    export_lambda_logs(pkl_path, json_path)


def worker(gpu_id, task_queue):
    while not task_queue.empty():
        try:
            task_name, command, log_dir, log_file_name = task_queue.get(timeout=1)
            run_command(command, gpu_id, task_name, log_dir, log_file_name)
            task_queue.task_done()
        except queue.Empty:
            break


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


def gpu_monitor(task_queue, active_gpus, threads, poll_interval=60):
    while not task_queue.empty():
        for gpu_id in get_free_gpu_ids():
            if gpu_id not in active_gpus and not task_queue.empty():
                print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} is now free — spawning worker.")
                active_gpus.add(gpu_id)
                t = threading.Thread(target=worker, args=(gpu_id, task_queue))
                t.start()
                threads.append(t)
        time.sleep(poll_interval)


def main():
    betas = [0.5, 0.3]
    tasks = []

    for beta in betas:
        log_dir = f'./logs/flex_dora_experiments/beta_{beta}/'
        os.makedirs(log_dir, exist_ok=True)

        log_file_name = f'Beta{beta}_FlexDoRA'
        base_cmd = (
            f'python3 main.py '
            f'--model qwen --dataset ag_news '
            f'--round 10 --epochs 1 '
            f'--n_clients 100 --sample_fraction 0.05 --seed 42 '
            f'--freeze_layer 0 --freeze_layers "" --finetune_epochs "" '
            f'--optimizer adam --lr 1e-4 --scheduler cosine '
            f'--beta {beta} --partition noniid '
            f'--peft dora --flex_dora --ft_classifier '
            f'--lora_r 8 --lora_alpha 16 '
            f'--dora_m_wd 0.0 --dora_m_lr 1e-4 '
            f'--logdir {log_dir} --log_file_name {log_file_name}'
        )

        tasks.append((f'Beta{beta}_FlexDoRA', base_cmd, log_dir, log_file_name))

    task_queue = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    initial_gpus = get_free_gpu_ids()
    if not initial_gpus:
        print(f"[{time.strftime('%H:%M:%S')}] No free GPUs found. Waiting...")
        while not initial_gpus:
            time.sleep(30)
            initial_gpus = get_free_gpu_ids()

    print(f"[{time.strftime('%H:%M:%S')}] Starting {len(tasks)} tasks on GPU(s): {initial_gpus}\n")

    active_gpus = set(initial_gpus)
    threads = []
    for gpu_id in initial_gpus:
        t = threading.Thread(target=worker, args=(gpu_id, task_queue))
        t.start()
        threads.append(t)

    monitor = threading.Thread(
        target=gpu_monitor,
        args=(task_queue, active_gpus, threads),
        daemon=True
    )
    monitor.start()

    for t in threads:
        t.join()

    print(f"[{time.strftime('%H:%M:%S')}] All experiments completed.")


if __name__ == "__main__":
    main()
