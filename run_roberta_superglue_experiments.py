import os
import subprocess
import time
import queue
import threading

def wait_for_gpu_free_by_vram(gpu_id, vram_threshold_mb=5000, check_interval=300):
    """지정된 GPU의 VRAM 사용량이 임계값 이하로 떨어질 때까지 대기합니다. 기본 5분(300초) 간격 체크"""
    print(f"[{time.strftime('%H:%M:%S')}] Checking if GPU {gpu_id} VRAM is below {vram_threshold_mb} MB...")
    first_wait = True
    while True:
        try:
            # nvidia-smi를 사용하여 해당 GPU의 VRAM 사용량(MiB 단위)을 가져옴
            output = subprocess.check_output(
                f"nvidia-smi -i {gpu_id} --query-gpu=memory.used --format=csv,noheader,nounits",
                shell=True,
                text=True
            )
            vram_used = int(output.strip())
            
            if vram_used < vram_threshold_mb:
                if not first_wait:
                    print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} VRAM ({vram_used} MB) is now below {vram_threshold_mb} MB. Starting next task...")
                    print("=" * 60)
                break
            
            if first_wait:
                print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu_id} is currently BUSY (VRAM: {vram_used} MB). Waiting...")
                first_wait = False
            
            # 5분 대기 (요청사항 반영)
            time.sleep(check_interval)
        except subprocess.CalledProcessError:
            print(f"[{time.strftime('%H:%M:%S')}] Failed to get nvidia-smi for GPU {gpu_id}. Retrying later.")
            time.sleep(check_interval)

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
            
        # 해당 GPU의 VRAM이 확보될 때까지 대기 (5분 주기로 체크)
        wait_for_gpu_free_by_vram(gpu_id, vram_threshold_mb=5000, check_interval=300)
        
        run_command(command, gpu_id, task_name, log_dir)
        task_queue.task_done()

def main():
    
    # 사용할 GPU 리스트
    gpus = [0, 1]
    
    # 데이터셋: SuperGLUE 4개 태스크
    datasets = ["superglue_boolq", "superglue_multirc", "superglue_record", "superglue_wic"]
    
    # 공통 하이퍼파라미터
    model = "roberta-large"
    rounds = 30
    epochs = 1
    n_clients = 3
    sample_fraction = 1.0
    beta = 0.5

    # 5개의 실험 메소드 정의 (FL+DoRA 계열 주석 처리)
    METHODS = {
        'FedIT':                    '--peft lora --trainable_A',
        # 'FL+DoRA(FlexLoRA)':        '--peft dora --flex_lora',
        # 'FL+DoRA(FlexLoRA+FFALoRA)':'--peft dora --flex_lora --flex_lora_freeze_a',
        'FeDoRA':                   '--peft dora --flex_lora --flex_lora_svd_a',
        'FlexLoRA':                 '--peft lora --flex_lora',
        'FFA-LoRA':                 '--peft lora --flex_lora --flex_lora_freeze_a',
        'FedEx-LoRA':               '--peft lora --fedex_lora --trainable_A',
    }

    log_base_dir = f"./logs/roberta_superglue_experiments/"
    
    task_queue = queue.Queue()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    seeds = [43, 44]
    
    for dataset in datasets:
        log_dir = os.path.join(log_base_dir, dataset)
        os.makedirs(log_dir, exist_ok=True)
        
        for seed in seeds:
            common_args = (
                f"--model {model} --dataset {dataset} --round {rounds} "
                f"--epochs {epochs} --n_clients {n_clients} --sample_fraction {sample_fraction} --seed {seed} "
                f"--optimizer adamw --lr 1e-4 --scheduler cosine --beta {beta} --partition noniid "
                f"--lora_r 8 --lora_alpha 8 --local_steps 50 --ft_classifier "
            )

            for method_name, method_args in METHODS.items():
                task_name = f"{dataset}_{method_name}_seed{seed}"
                cmd = f"cd {script_dir} && python3 main.py {common_args} {method_args} --logdir {log_dir} --log_file_name '{task_name}'"
                task_queue.put((task_name, cmd, log_dir))

    total_tasks = task_queue.qsize()
    print(f"[{time.strftime('%H:%M:%S')}] Starting {total_tasks} SuperGLUE experiments across {len(gpus)} GPUs (GPUs: {gpus})")
    print(f"Datasets: {datasets}")
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
    
    print(f"[{time.strftime('%H:%M:%S')}] All {total_tasks} SuperGLUE experiments have been completed.")

if __name__ == "__main__":
    main()
