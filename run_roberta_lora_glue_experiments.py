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
            print(f"[{task_name}|GPU{gpu_id}] {line}", end="")
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
    # 사용할 GPU 리스트
    gpus = [0, 1]
    
    # 사용할 데이터셋 리스트
    datasets = ["glue_sst2", "glue_mnli", "glue_qqp"]
    
    # 공통 하이퍼파라미터
    model = "roberta-large"
    rounds = 30
    epochs = 1
    n_clients = 3
    sample_fraction = 1.0
    beta = 0.5

    METHODS = {
        'FlexLoRA':     '--peft lora --flex_lora',
        'FFA-LoRA':     '--peft lora --flex_lora --flex_lora_freeze_a',
        'FedEx-LoRA':   '--peft lora --fedex_lora --trainable_A',
    }

    task_queue = queue.Queue()
    for dataset in datasets:
        log_dir = f"./logs/roberta_lora_experiments/{dataset}/"
        
        # 공통 명령어
        common_args = (
            f"--model {model} --dataset {dataset} --round {rounds} "
            f"--epochs {epochs} --n_clients {n_clients} --sample_fraction {sample_fraction} --seed 42 "
            f"--optimizer adamw --lr 1e-4 --scheduler cosine --beta {beta} --partition noniid "
            f"--lora_r 8 --lora_alpha 8 --local_steps 50 --ft_classifier "
        )
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for method_name, method_args in METHODS.items():
            task_name = f"{dataset}_{method_name}"
            cmd = f"cd {script_dir} && python3 main.py {common_args} {method_args} --logdir {log_dir} --log_file_name {task_name}"
            task_queue.put((task_name, cmd, log_dir))

    total_tasks = task_queue.qsize()
    print(f"[{time.strftime('%H:%M:%S')}] Starting {total_tasks} experiments across {len(gpus)} GPUs (GPUs: {gpus})")
    print(f"Datasets to run: {datasets}")
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