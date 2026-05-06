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

    lambda_data = data.get('flex_lora_lambda', [])
    if not lambda_data:
        print(f"  [warn] flex_lora_lambda is empty in {pkl_path}")
        return

    serializable = []
    for round_logs in lambda_data:
        round_dict = {}
        for layer, vals in round_logs.items():
            round_dict[layer] = {
                k: v.tolist() if isinstance(v, torch.Tensor) else float(v)
                for k, v in vals.items()
            }
        serializable.append(round_dict)

    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"  [info] lambda_logs saved → {json_path} ({len(serializable)} rounds)")


def run_command(command, gpu_id, task_name, log_dir, log_file_name):
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
            bufsize=1
        )

        for line in process.stdout:
            f.write(line)
            f.flush()

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


def main():
    betas = [0.5, 0.3]
    tasks = []

    for beta in betas:
        log_dir = f'./logs/flex_lora_experiments/beta_{beta}/'
        os.makedirs(log_dir, exist_ok=True)

        log_file_name = f'Beta{beta}_FlexLoRA'
        base_cmd = (
            f'python3 main.py '
            f'--model qwen --dataset ag_news '
            f'--round 10 --epochs 1 '
            f'--n_clients 100 --sample_fraction 0.05 --seed 42 '
            f'--freeze_layer 0 --freeze_layers "" --finetune_epochs "" '
            f'--optimizer adam --lr 1e-4 --scheduler cosine '
            f'--beta {beta} --partition noniid '
            f'--peft dora --flex_lora '
            f'--lora_r 8 --lora_alpha 16 '
            f'--dora_m_wd 0.0 --dora_m_lr 1e-4 '
            f'--logdir {log_dir} --log_file_name {log_file_name}'
        )

        tasks.append((f'Beta{beta}_FlexLoRA', base_cmd, log_dir, log_file_name))

    task_queue = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    gpus = [0, 1]
    threads = []

    print(f"[{time.strftime('%H:%M:%S')}] Starting {len(tasks)} tasks across {len(gpus)} GPUs...\n")
    for gpu_id in gpus:
        t = threading.Thread(target=worker, args=(gpu_id, task_queue))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"[{time.strftime('%H:%M:%S')}] All experiments completed.")


if __name__ == "__main__":
    main()
