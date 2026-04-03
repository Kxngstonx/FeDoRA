import os
import subprocess
import time
import queue
import threading
import glob
import json

def extract_params(filename, is_cosine=False):
    params = {}
    base_name = os.path.basename(filename).replace('.json', '')
    parts = base_name.split('_')
    for p in parts:
        if p.startswith('r') and p[1:].isdigit(): params['r'] = p[1:]
        elif p.startswith('a') and p[1:].isdigit(): params['a'] = p[1:]
        elif p.startswith('wd'): params['wd'] = p[2:]
        elif p.startswith('lr'): params['lr'] = p[2:]
        
    if is_cosine:
        for p in parts:
            if p.startswith('t'): params['tau'] = p[1:]
            elif p.startswith('g'): params['gamma'] = p[1:]
    else:
        for p in parts:
            if p.startswith('rat'): params['rat'] = p[3:]
            elif p.startswith('mul'): params['mul'] = p[3:]
    return params

def get_best_experiment(prefix, log_dir='./logs/decoupled_experiments/'):
    best_acc = 0.0
    best_file = None
    if not os.path.exists(log_dir):
        return None, 0.0
    
    for f in glob.glob(os.path.join(log_dir, f'{prefix}_*.json')):
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                acc = max(data.get('local_acc', [0.0]))
                if acc > best_acc:
                    best_acc = acc
                    best_file = f
        except Exception:
            pass
    return best_file, best_acc

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
            bufsize=1
        )
        
        for line in process.stdout:
            f.write(line)
            f.flush()
            
        process.wait()
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Finished {task_name} on GPU {gpu_id}.")

def worker(gpu_id, task_queue):
    while not task_queue.empty():
        try:
            task_name, command, log_dir = task_queue.get(timeout=1)
            run_command(command, gpu_id, task_name, log_dir)
            task_queue.task_done()
        except queue.Empty:
            break

def main():
    best_cos_file, _ = get_best_experiment("Cos")
    best_warm_file, _ = get_best_experiment("Warm")

    cos_params = extract_params(best_cos_file, True) if best_cos_file else {'r': '8', 'a': '16', 'wd': '0.0', 'lr': '1e-4', 'tau': '1.0', 'gamma': '0.0'}
    warm_params = extract_params(best_warm_file, False) if best_warm_file else {'r': '8', 'a': '16', 'wd': '0.0', 'lr': '1e-4', 'rat': '0.1', 'mul': '2.0'}

    best_lr = cos_params.get('lr', '1e-4')
    
    betas = [0.5, 0.3, 0.1]
    tasks = []
    
    for beta in betas:
        BETA_LOG_DIR = f'./logs/fixed_a_experiments/beta_{beta}/'
        os.makedirs(BETA_LOG_DIR, exist_ok=True)
        
        beta_base = f'python3 main.py --model qwen --dataset ag_news --round 50 --epochs 1 --n_clients 100 --sample_fraction 0.05 --seed 42 --freeze_layer 0 --freeze_layers "" --finetune_epochs "" --optimizer adam --lr {best_lr} --scheduler cosine --beta {beta} --partition noniid'

        if beta == 0.1:
            tasks.append((f"Beta{beta}_Full_FT", f"{beta_base} --peft none --logdir {BETA_LOG_DIR} --log_file_name Beta{beta}_Full_FT", BETA_LOG_DIR))

        tasks.extend([
            (f"Beta{beta}_LoRA", f"{beta_base} --peft lora --lora_r {cos_params.get('r', '8')} --lora_alpha {cos_params.get('a', '16')} --logdir {BETA_LOG_DIR} --log_file_name Beta{beta}_LoRA", BETA_LOG_DIR),
            (f"Beta{beta}_DoRA_Coupled", f"{beta_base} --peft dora --lora_r {cos_params.get('r', '8')} --lora_alpha {cos_params.get('a', '16')} --dora_m_wd 0.0 --dora_m_lr {best_lr} --logdir {BETA_LOG_DIR} --log_file_name Beta{beta}_DoRA_Coupled", BETA_LOG_DIR),
            (f"Beta{beta}_DoRA_Decoupled", f"{beta_base} --peft dora --decoupled_dora True --lora_r {cos_params.get('r', '8')} --lora_alpha {cos_params.get('a', '16')} --dora_m_wd 0.0 --dora_m_lr {best_lr} --logdir {BETA_LOG_DIR} --log_file_name Beta{beta}_DoRA_Decoupled", BETA_LOG_DIR),
            (f"Beta{beta}_Best_Cosine", f"{beta_base} --peft dora --decoupled_dora True --use_cosine_recal --dora_cos_tau {cos_params.get('tau', '1.0')} --dora_cos_gamma {cos_params.get('gamma', '0.0')} --lora_r {cos_params.get('r', '8')} --lora_alpha {cos_params.get('a', '16')} --dora_m_wd {cos_params.get('wd', '0.0')} --dora_m_lr {cos_params.get('lr', '1e-4')} --logdir {BETA_LOG_DIR} --log_file_name Beta{beta}_Best_Cosine", BETA_LOG_DIR),
            (f"Beta{beta}_Best_Warmup", f"{beta_base} --peft dora --decoupled_dora True --dora_warmup_ratio {warm_params.get('rat', '0.1')} --dora_warmup_lr_mult {warm_params.get('mul', '2.0')} --lora_r {warm_params.get('r', '8')} --lora_alpha {warm_params.get('a', '16')} --dora_m_wd {warm_params.get('wd', '0.0')} --dora_m_lr {warm_params.get('lr', '1e-4')} --logdir {BETA_LOG_DIR} --log_file_name Beta{beta}_Best_Warmup", BETA_LOG_DIR)
        ])

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
