import os
import subprocess
import time
import queue
import threading

def wait_for_gpu_free(gpu_id):
    """지정된 GPU에 활성 학습 프로세스(compute process)가 없을 때까지 대기합니다."""
    print(f"[{time.strftime('%H:%M:%S')}] Checking if GPU {gpu_id} is free of active training processes...")
    first_wait = True
    while True:
        try:
            # nvidia-smi를 사용하여 해당 GPU에서 실행 중인 compute 프로세스의 PID를 가져옴
            output = subprocess.check_output(
                f"nvidia-smi -i {gpu_id} --query-compute-apps=pid --format=csv,noheader",
                shell=True,
                text=True
            )
            pids = [line.strip() for line in output.strip().split('\n') if line.strip()]
            
            if not pids:
                if not first_wait:
                    print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} is now FREE. Starting next task...")
                    print("=" * 60)
                break
            
            if first_wait:
                print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} is currently BUSY (running PIDs: {pids}). Waiting...")
                first_wait = False
            
            time.sleep(30)
        except subprocess.CalledProcessError:
            # nvidia-smi 에러 발생 시 루프 탈출
            break

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
    while True:
        try:
            task_name, command, log_dir = task_queue.get_nowait()
        except queue.Empty:
            break
            
        # 해당 GPU가 비어있을 때까지 대기
        wait_for_gpu_free(gpu_id)
        
        run_command(command, gpu_id, task_name, log_dir)
        task_queue.task_done()

def main():
    
    # 사용할 GPU 리스트
    gpus = [0, 1]
    
    # 데이터셋: ANLI (R1+R2+R3 통합 train, 각 라운드별 test)
    dataset = "anli"
    
    # 공통 하이퍼파라미터 (기존 GLUE 실험과 동일)
    model = "roberta-large"
    rounds = 30
    epochs = 1
    n_clients = 3
    sample_fraction = 1.0
    beta = 0.5

    # 7개의 실험 메소드 정의 (새 명칭)
    METHODS = {
        # DoRA 기반 (run_roberta_glue_experiments.py 계열)
        'FedIT':                    '--peft lora --trainable_A',
        'FL+DoRA(FlexLoRA)':        '--peft dora --flex_lora',
        'FL+DoRA(FlexLoRA+FFALoRA)':'--peft dora --flex_lora --flex_lora_freeze_a',
        'FeDoRA':                   '--peft dora --flex_lora --flex_lora_svd_a',
        # LoRA 기반 (run_roberta_lora_glue_experiments.py 계열)
        'FlexLoRA':                 '--peft lora --flex_lora',
        'FFA-LoRA':                 '--peft lora --flex_lora --flex_lora_freeze_a',
        'FedEx-LoRA':               '--peft lora --fedex_lora --trainable_A',
    }

    # 로그 디렉토리
    log_dir = f"./logs/roberta_anli_experiments/{dataset}/"
    os.makedirs(log_dir, exist_ok=True)

    # 공통 명령어
    common_args = (
        f"--model {model} --dataset {dataset} --round {rounds} "
        f"--epochs {epochs} --n_clients {n_clients} --sample_fraction {sample_fraction} --seed 42 "
        f"--optimizer adamw --lr 1e-4 --scheduler cosine --beta {beta} --partition noniid "
        f"--lora_r 8 --lora_alpha 8 --local_steps 50 --ft_classifier "
    )

    task_queue = queue.Queue()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for method_name, method_args in METHODS.items():
        task_name = f"{dataset}_{method_name}"
        cmd = f"cd {script_dir} && python3 main.py {common_args} {method_args} --logdir {log_dir} --log_file_name '{task_name}'"
        task_queue.put((task_name, cmd, log_dir))

    total_tasks = task_queue.qsize()
    print(f"[{time.strftime('%H:%M:%S')}] Starting {total_tasks} ANLI experiments across {len(gpus)} GPUs (GPUs: {gpus})")
    print(f"Dataset: {dataset}")
    print(f"Methods: {list(METHODS.keys())}")
    print("=" * 60)

    # 워커 스레드 생성 (각 GPU당 1개의 스레드가 1개의 프로세스를 순차 실행)
    threads = []
    for gpu_id in gpus:
        t = threading.Thread(target=worker, args=(gpu_id, task_queue))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
    
    print(f"[{time.strftime('%H:%M:%S')}] All {total_tasks} ANLI experiments have been completed.")

if __name__ == "__main__":
    main()
