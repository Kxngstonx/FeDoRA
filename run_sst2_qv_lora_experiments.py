import os
import subprocess
import time
import queue
import threading

def run_command(command, gpu_id, task_name, log_dir):
    """지정된 GPU에서 단일 명령어를 실행하고 로그를 파일에 기록합니다."""
    print(f"[{time.strftime('%H:%M:%S')}] Starting '{task_name}' on GPU {gpu_id}...\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    log_file_path = os.path.join(log_dir, f"{task_name}.log")
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

    if process.returncode == 0:
        print(f"[{time.strftime('%H:%M:%S')}] Successfully finished '{task_name}' on GPU {gpu_id}.")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Task '{task_name}' failed on GPU {gpu_id} with return code {process.returncode}.")
    print("-" * 60)

def worker(gpu_id, task_queue):
    while not task_queue.empty():
        try:
            task_name, command, log_dir = task_queue.get(timeout=1)
            run_command(command, gpu_id, task_name, log_dir)
            task_queue.task_done()
        except queue.Empty:
            break

def main():
    gpus = [0, 1]

    model            = "roberta-large"
    dataset          = "glue_sst2"
    rounds           = 20
    epochs           = 1
    local_steps      = 10       # step 수가 10을 초과하면 조기 중단
    n_clients        = 10
    sample_fraction  = 1.0      # 매 라운드 전체 클라이언트 참여
    beta             = 0.5      # Dirichlet 파라미터
    batch_size       = 32
    lr               = 2e-4
    lora_r           = 8
    lora_alpha       = 32       # scaling factor
    lora_dropout     = 0.05
    # LoRA 적용 대상: Query, Value 모듈만 / 나머지(Key, FFN 등)는 Full Fine-Tune
    lora_target_modules = "query,value"

    METHODS = {
        'LoRA':           '--peft lora',
        'FFALoRA':'--peft lora --trainable_A',
        'DoRA':           '--peft dora',
        'FlexDoRA':       '--peft dora --flex_dora',
        'FFADoRA':        '--peft dora --flex_dora --flex_dora_freeze_a',
    }

    log_dir = f"./logs/sst2_qv_lora_experiments/"
    os.makedirs(log_dir, exist_ok=True)

    common_args = (
        f"--model {model} --dataset {dataset} --round {rounds} "
        f"--epochs {epochs} --local_steps {local_steps} "
        f"--n_clients {n_clients} --sample_fraction {sample_fraction} --seed 42 "
        f"--optimizer adamw --lr {lr} --scheduler exponential "
        f"--beta {beta} --partition noniid "
        f"--batch_size {batch_size} "
        f"--lora_r {lora_r} --lora_alpha {lora_alpha} --lora_dropout {lora_dropout} "
        f"--lora_target_modules {lora_target_modules} "
        f"--ft_classifier "
    )

    task_queue = queue.Queue()

    for method_name, method_args in METHODS.items():
        task_name = f"sst2_{method_name}"
        cmd = f"python3 main.py {common_args} {method_args} --logdir {log_dir} --log_file_name {task_name}"
        task_queue.put((task_name, cmd, log_dir))

    total_tasks = task_queue.qsize()
    print(f"[{time.strftime('%H:%M:%S')}] Starting {total_tasks} experiments across {len(gpus)} GPUs (GPUs: {gpus})")
    print(f"Dataset: {dataset} | Model: {model} | Rounds: {rounds} | Clients: {n_clients}")
    print("=" * 60)

    threads = []
    for gpu_id in gpus:
        t = threading.Thread(target=worker, args=(gpu_id, task_queue))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"[{time.strftime('%H:%M:%S')}] All {total_tasks} experiments have been completed.")

if __name__ == "__main__":
    main()
