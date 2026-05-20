import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import copy
import time
import os
import gc
import json
import pickle

from utils import *
from utils import prepare_data_for_training
from peft_utils import inject_peft
import train


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--datadir', default="./data/", help="Data directory")
    parser.add_argument('--logdir', default="./logs/", help='Log directory path')
    parser.add_argument('--log_file_name', default=None, help='log file name')
    parser.add_argument('--device', default='cuda:0', help='The device to run the program')
    parser.add_argument('--num_workers', default=0, type=int, help='the number of workers for each dataloader')

    parser.add_argument('--dataset', default='ag_news', help='dataset used for training')
    parser.add_argument('--model', default='qwen', help='model architecture')
    parser.add_argument('--group_norm', action='store_true', help='replace batch_norm with group_norm')
    parser.add_argument('--num_groups', type=int, default=32, help='num of groups in group_norm')
    parser.add_argument('--in_channels', type=int, default=3)

    parser.add_argument('--optimizer', default='sgd', help='the optimizer')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum for the optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=32, help='batch size for validation or test')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--scheduler', default='step', help='scheduler for rounds [linear, cosine, step, None]')
    parser.add_argument('--schedule_round', type=int, nargs='+', default=[250])
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--eta_min', type=float, default=0.0, help='minimum learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--local_steps', type=int, default=0, help='local gradient update steps per round (0 = epoch-based)')
    parser.add_argument('--reg', type=float, default=1e-3, help="L2 regularization strength")

    parser.add_argument('--partition', default='noniid', help='iid, noniid, noniid_balanced')
    parser.add_argument('--alg', default='fedavg', help='communication strategy')
    parser.add_argument('--round', type=int, default=50, help='number of maximum communication round')
    parser.add_argument('--n_clients', type=int, default=100, help='number of workers in a distributed cluster')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution')
    parser.add_argument('--unavailability', default='stationary')
    parser.add_argument('--min_require_size', type=int, default=1)
    parser.add_argument('--sample_fraction', type=float, default=0.1)
    parser.add_argument('--time', type=int, default=0)

    parser.add_argument('--mu', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--server_momentum', type=float, default=0)
    parser.add_argument('--feddecorr', action='store_true')
    parser.add_argument('--feddecorr_coef', type=float, default=0.01)
    parser.add_argument('--use_projection_head', type=int, default=0)
    parser.add_argument('--lambda_pull', type=float, default=0.5)
    parser.add_argument('--lambda_push', type=float, default=0.5)
    parser.add_argument('--server_adam', type=float, default=0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--server_yogi', type=float, default=0)

    parser.add_argument('--freeze_layer', type=int, default=0)
    parser.add_argument('--freeze_layers', type=str, default='')
    parser.add_argument('--freeze_block_pos', type=str, choices=['first', 'last'], default='first')
    parser.add_argument('--freeze_classifier', action='store_true')
    parser.add_argument('--finetune_epochs', type=str, default='')

    parser.add_argument('--peft', type=str, choices=['none', 'lora', 'dora'], default='none')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='Dropout probability for LoRA/DoRA layers')
    parser.add_argument('--dora_m_lr', type=float, default=None, help='Specific learning rate for DoRA m parameter')
    parser.add_argument('--dora_m_wd', type=float, default=None, help='Specific weight decay for DoRA m parameter')
    parser.add_argument('--decoupled_dora', type=bool, default=False, help='If True, keep DoRA magnitude m local and do not aggregate it.')
    parser.add_argument('--dora_cos_tau', type=float, default=1.0, help='Penalty sensitivity for cosine re-calibration')
    parser.add_argument('--dora_cos_gamma', type=float, default=0.0, help='Minimum preservation threshold for cosine re-calibration')
    parser.add_argument('--dora_warmup_ratio', type=float, default=0.0, help='Ratio of local steps to freeze A, B and only train m')
    parser.add_argument('--dora_warmup_lr_mult', type=float, default=1.0, help='LR multiplier for m during warmup')
    parser.add_argument('--use_cosine_recal', action='store_true', help='Enable Cosine Similarity Re-calibration')
    parser.add_argument('--flex_lora', action='store_true', help='FlexLoRA: train lora_A on clients, use SVD-based server aggregation.')
    parser.add_argument('--flex_lora_freeze_a', action='store_true', help='FlexLoRA: freeze A on clients, use Least Squares (not SVD) for B_new on server.')
    parser.add_argument('--ft_classifier', action='store_true', help='Full fine-tune the final classifier layer instead of applying LoRA/DoRA.')
    parser.add_argument('--trainable_A', action='store_true', help='Make lora_A trainable in LoRA/DoRA (non-FlexLoRA mode)')


    args = parser.parse_args()
    return args


# 특정 Layer의 Parameters를 freeze
def apply_layer_freezing(net, args):
    if args.alg == 'fedavg':
        layers_to_freeze = []
        if getattr(args, 'freeze_layers_list', None): # Layer list
            layers_to_freeze = args.freeze_layers_list
        elif args.freeze_layer > 0: # One layer
            layers_to_freeze = [args.freeze_layer]
        if layers_to_freeze:
            for L in layers_to_freeze: # Layer는 여러 Residual Block으로 구성, 각 Block 안에는 여러 Conv Layer 존재
                layer_name = f'layer{L}'
                if hasattr(net, layer_name):
                    layer_module = getattr(net, layer_name)
                    mode = getattr(args, 'freeze_block_pos', 'first')
                    if mode == 'first': # 각 Block의 conv1
                        for block in layer_module:
                            if hasattr(block, 'conv1'):
                                for param in block.conv1.parameters(): 
                                    param.requires_grad = False # Freeze
                    elif mode == 'last':
                        for block in layer_module:
                            if hasattr(block, 'conv3'):
                                for param in block.conv3.parameters():
                                    param.requires_grad = False
                            elif hasattr(block, 'conv2'):
                                for param in block.conv2.parameters():
                                    param.requires_grad = False
                    else:
                        for block in layer_module:
                            if hasattr(block, 'conv1'):
                                for param in block.conv1.parameters():
                                    param.requires_grad = False

    if args.alg in ('fedavg', 'fedbabu') and args.freeze_classifier: # BABU: Body-As-Base, Un-trained head = Classifier는 Freeze, Feature Extractor(Body)만 학습
        try:
            classifier_name = list(net.named_children())[-1][0]
        except Exception:
            classifier_name = None
        if classifier_name is not None:
            for name, param in net.named_parameters():
                if name.startswith(classifier_name + '.'):
                    param.requires_grad = False


def main():
    args = get_args()
    
    if args.freeze_layers:
        args.freeze_layers_list = [int(x) for x in args.freeze_layers.split(',') if x.strip().isdigit()]
    else:
        args.freeze_layers_list = []

    if args.finetune_epochs:
        args.finetune_epochs_list = [int(x) for x in args.finetune_epochs.split(',') if x.strip().isdigit()]
    else:
        args.finetune_epochs_list = []

    if args.alg == 'fedbabu':
        args.freeze_classifier = True
        if getattr(args, 'freeze_layers_list', None) and len(args.freeze_layers_list) > 0:
            args._fedbabu_ignored_freeze_layers = args.freeze_layers_list
            args.freeze_layers_list = []

    if args.time > 0:
        time.sleep(args.time)
    device = torch.device(args.device)
    logger, log_file_name = init_logger(args)

    if getattr(args, '_fedbabu_ignored_freeze_layers', None):
        logger.warning(f"FedBABU ignores --freeze_layers; ignoring {args._fedbabu_ignored_freeze_layers} and freezing only the classifier/head.")

    # Client별 Non-IID Data 분할, DataLoader 생성
    (global_train_dataset, global_val_dataset, global_test_dataset,
     client_data_map, clients_at_rounds, client_datasets,
     global_train_dataloader, global_val_dataloader, global_test_dataloader,
     client_dataloaders, client_test_dataloaders, client_finetune_train) = prepare_data_for_training(args, logger)

    if args.alg == 'fedsea':
        fedsea_datasets, fedsea_dataloaders = get_fedsea_dataloaders(client_datasets, args)

    # Initialize SINGLE global model and ONE local model to save GPU memory
    base_global_model = init_nets(global_train_dataset, 1, args, device, base=True, use_projection_head=args.use_projection_head)[0]
    _ft_classifier_global_skip = []
    if args.peft != 'none' and getattr(args, 'ft_classifier', False):
        if 'roberta' in args.model:
            _ft_classifier_global_skip = ['classifier']
        elif 'qwen' in args.model:
            _ft_classifier_global_skip = ['score']
    _peft_kwargs = dict(trainable_A=(getattr(args, 'flex_lora', False) and not getattr(args, 'flex_lora_freeze_a', False)) or getattr(args, 'trainable_A', False), global_skip_modules=_ft_classifier_global_skip)
    base_global_model = inject_peft(base_global_model, args.peft, args.lora_r, args.lora_alpha, getattr(args, 'lora_dropout', 0.0), **_peft_kwargs)

    global_model = copy.deepcopy(base_global_model)
    local_model  = copy.deepcopy(base_global_model)

    base_w = base_global_model.state_dict()
    global_model.load_state_dict(base_w, strict=False)

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v: moment_v[key] = 0
    if args.server_adam or args.server_yogi:
        moment_m = {key: torch.zeros_like(param) for key, param in global_model.state_dict().items()}
        moment_v = {key: torch.zeros_like(param) for key, param in global_model.state_dict().items()}
        t_adam = 0
        t_yogi = 0

    client_discriminator = None
    feature_mask = None
    client_iid_generators = {}
    if args.alg == 'fedsea':
        client_discriminator, optimizer_discriminator, optimizer_attention, attention_p, feature_dim = init_fedsea(global_model, global_test_dataloader, client_iid_generators, args)

    pkl_dict = {'args': vars(args), 'avg_train_loss': [], 'acc': []}
    lr = args.lr

    from peft_utils import (get_dora_components, get_dora_delta_scalars,
                            get_client_dora_delta_scalars, compute_temporal_dora_correlation)
    initial_dora_components = get_dora_components(global_model) if args.peft == 'dora' else None
    dora_server_dm_accum = []
    dora_server_dv_accum = []
    checkpoint_interval = max(1, args.round // 5)

    
    client_m_storage = {} # Store specific m per client
    client_v_storage = {} # Store specific V per client for cosine

    for round in range(args.round):
        logger.info(f'round:{round}')
        clients_this_round = clients_at_rounds[round]

        global_model.eval()
        for param in global_model.parameters(): param.requires_grad = False
        global_w = global_model.state_dict()
        
        dataloaders_this_round = {i: client_dataloaders[i] for i in clients_this_round}

        if args.server_momentum or args.server_adam or args.server_yogi:
            old_w = copy.deepcopy(global_w)

        global_class_center = None
        # omitted fedproc/fedsea global_class_center logic for brevity in this memory optimized version if unused
        
        fedsea_params = {}

        # SEQUENTIAL TRAINING TO SAVE MEMORY
        t0 = time.time()
        round_loss = 0.0
        updated_client_weights = {}
        personalized_accs = []
        
        for cid in clients_this_round:
            
            local_model.load_state_dict(global_w, strict=False)
            
            # Restore local m for decoupled DoRA (flex_lora 활성 시 서버 값을 그대로 사용하므로 복원 안 함)
            if args.peft == 'dora' and args.decoupled_dora and not getattr(args, 'flex_lora', False):
                if cid in client_m_storage:
                    local_model.load_state_dict(client_m_storage[cid], strict=False) # Client의 magnitude parameter 집계에서 제외
                    
            # --- DoRA Cosine Re-calibration Logic ---
            if getattr(args, 'use_cosine_recal', False) and cid in client_v_storage:
                for name, module in local_model.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'm') and name in client_v_storage[cid]:
                        W0 = module.linear.weight if hasattr(module, 'linear') else module.conv.weight
                        # Current global direction V_new
                        BA_global = (module.lora_B @ module.lora_A) * module.scaling
                        V_new_flat = (W0 + BA_global).flatten()
                        
                        # Old local direction V_old
                        A_old, B_old = client_v_storage[cid][name]
                        BA_old = (B_old @ A_old) * module.scaling
                        V_old_flat = (W0 + BA_old).flatten()
                        
                        # Cosine Sim [-1,1]
                        cos_sim = torch.nn.functional.cosine_similarity(V_old_flat, V_new_flat, dim=0)
                        
                        # Apply Gamma and Tau
                        adjusted_cos = torch.max(torch.tensor(args.dora_cos_gamma, device=device), cos_sim) # gamma = Penalty의 Lower bound, cos_sim이 음수일 때 penalty가 0이 되는 것을 방지
                        if adjusted_cos < 0:
                            adjusted_cos = torch.tensor(args.dora_cos_gamma, device=device)
                        penalty = torch.pow(adjusted_cos, args.dora_cos_tau) # tau = Penalty의 degree
                        
                        # Scale m down safely
                        with torch.no_grad():
                            module.m.data *= penalty
            # ----------------------------------------

            apply_layer_freezing(local_model, args)

            if args.peft == 'dora':
                before_dora_components = get_dora_components(local_model)

            nets_dict = {cid: local_model}
            dl_dict = {cid: dataloaders_this_round[cid]}

            # ensure adjust_lr doesn't double dip by passing a flag or letting it run
            args.last_decay_round = -1 # prevent multiple decays inside same round

            loss, new_lr = train.train_local_net(
                dataloaders=dl_dict, nets=nets_dict, global_model=global_model, prev_nets=None,
                device=device, round=round, lr=lr, args=args, logger=logger,
                global_class_center=global_class_center, **fedsea_params
            )
            lr = new_lr

            if args.peft == 'dora':
                after_dora_components = get_dora_components(local_model)
                dm_layers, dv_layers = get_client_dora_delta_scalars(before_dora_components, after_dora_components)
                client_corr = compute_temporal_dora_correlation(dm_layers, dv_layers)
                logger.info(f'ROUND {round} CLIENT {cid} DoRA Corr (layer-wise): {client_corr:.4f}')
                pkl_dict.setdefault('client_dora_corr', []).append({'round': round, 'cid': cid, 'corr': client_corr})

            round_loss += loss
            
            # calc local acc
            try:
                test_loader = client_test_dataloaders.get(cid, global_test_dataloader)
                per_acc = compute_accuracy(local_model, test_loader, device=device)
            except Exception:
                per_acc = 0.0
            personalized_accs.append(per_acc)
            
            # Extract weights to CPU
            updated_client_weights[cid] = {k: v.clone() for k, v in local_model.state_dict().items()}
            
            # Save local m for decoupled DoRA
            if args.peft == 'dora' and args.decoupled_dora and not getattr(args, 'flex_lora', False):
                client_m_storage[cid] = {k: v.clone() for k, v in local_model.state_dict().items() if k.endswith('.m')}
                
            # Save V_local for Cosine Re-calibration next round
            if getattr(args, 'use_cosine_recal', False):
                client_v_storage[cid] = {}
                for name, module in local_model.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'm'):
                        client_v_storage[cid][name] = (module.lora_A.detach().clone(), module.lora_B.detach().clone())

            
            # Free cache
            if device.type == 'cuda': torch.cuda.empty_cache()

        t1 = time.time()
        print(f'train time: {t1-t0}')
        avg_loss = round_loss / len(clients_this_round)
        pkl_dict['avg_train_loss'].append(avg_loss)

        mean_personalized = sum(personalized_accs) / len(personalized_accs) if len(personalized_accs) > 0 else 0.0
        pkl_dict.setdefault('local_acc', []).append(mean_personalized)
        logger.info('----------------------------------------------------------------------')
        logger.info(f'ROUND {round}')
        logger.info(f'--- MEAN_LOCAL_ACC = {mean_personalized:.4f}')
        logger.info('----------------------------------------------------------------------')

        # aggregation
        total_batches = sum([len(dataloaders_this_round[j]) for j in dataloaders_this_round])
        fed_avg_freqs = {j: len(dataloaders_this_round[j]) / total_batches for j in dataloaders_this_round}

        classifier_prefix = None
        if args.alg in ('fedavg', 'fedbabu') and args.freeze_classifier:
            try:
                classifier_prefix = list(global_model.named_children())[-1][0] + '.'
            except Exception: pass

        agg_layers_to_freeze = args.freeze_layers_list if getattr(args, 'freeze_layers_list', None) else ([args.freeze_layer] if args.freeze_layer > 0 else [])
        freeze_mode = getattr(args, 'freeze_block_pos', 'first')

        for idx, (net_id, net_para_dev) in enumerate(updated_client_weights.items()):
            net_para = {k: v for k, v in net_para_dev.items()}
            weight = fed_avg_freqs[net_id]
            if idx == 0:
                for key in net_para:
                    skip = False
                    for L in agg_layers_to_freeze: # Freeze Layer는 Aggregation skip
                        if key.startswith(f'layer{L}.'):
                            if freeze_mode == 'first' and '.conv1.' in key: skip = True
                            elif freeze_mode == 'last' and ('.conv3.' in key or '.conv2.' in key): skip = True
                    if classifier_prefix and key.startswith(classifier_prefix): skip = True
                    if args.decoupled_dora and key.endswith('.m'): skip = True
                    # FlexLoRA: DoRA 파라미터는 flex_lora_aggregate가 별도 처리
                    if getattr(args, 'flex_lora', False) and (key.endswith('.lora_A') or key.endswith('.lora_B') or key.endswith('.m')): skip = True
                    if not skip: global_w[key] = net_para[key] * weight
            else:
                for key in net_para:
                    skip = False
                    for L in agg_layers_to_freeze:
                        if key.startswith(f'layer{L}.'):
                            if freeze_mode == 'first' and '.conv1.' in key: skip = True
                            elif freeze_mode == 'last' and ('.conv3.' in key or '.conv2.' in key): skip = True
                    if classifier_prefix and key.startswith(classifier_prefix): skip = True
                    if args.decoupled_dora and key.endswith('.m'): skip = True
                    # FlexLoRA: DoRA 파라미터는 flex_lora_aggregate가 별도 처리
                    if getattr(args, 'flex_lora', False) and (key.endswith('.lora_A') or key.endswith('.lora_B') or key.endswith('.m')): skip = True
                    if not skip: global_w[key] += net_para[key] * weight

        global_model.load_state_dict(global_w, strict=False)

        # FlexLoRA 집계: W_k 재구성 → FedAvg → SVD 분해로 m_new, B_new, A_new 초기화
        if getattr(args, 'flex_lora', False) and args.peft == 'dora':
            from peft_utils import flex_lora_aggregate
            updated_dora_params, lambda_logs = flex_lora_aggregate(
                updated_client_weights, fed_avg_freqs, global_model,
                freeze_a=getattr(args, 'flex_lora_freeze_a', False))
            global_model.load_state_dict(updated_dora_params, strict=False)
            global_w.update(updated_dora_params)

            round_sv_logs = {}
            is_freeze_a = getattr(args, 'flex_lora_freeze_a', False)
            err_label = 'ls_err' if is_freeze_a else 'trunc_err'
            for lname, lvals in lambda_logs.items():
                err_str = (
                    f'ls_err={lvals["ls_err"]:.6f}, '
                    f'ls_err_relative={lvals["ls_err_relative"]:.6f}'
                    if is_freeze_a else
                    f'trunc_err={lvals["trunc_err"]:.6f}, '
                    f'trunc_err_relative={lvals["trunc_err_relative"]:.6f}'
                )
                logger.info(
                    f'ROUND {round} FlexLoRA [{lname}] '
                    f'lambda_diff={lvals["lambda_diff"]:.6f}, '
                    f'lambda_relative={lvals["lambda_relative"]:.6f} | '
                    f'w_diff={lvals["w_diff"]:.6f}, '
                    f'w_relative={lvals["w_relative"]:.6f}, '
                    f'{err_str}')
                round_sv_logs[lname] = lvals['singular_values']
            pkl_dict.setdefault('flex_lora_sv', []).append(round_sv_logs)

        if args.peft == 'dora':
            from utils import DoRASimilarityCalculator
            sim_calc = DoRASimilarityCalculator(temperature=1.0)
            
            server_dora = get_dora_components(global_model)
            dm_layers, dv_layers = get_dora_delta_scalars(initial_dora_components, server_dora)
            round_corr = compute_temporal_dora_correlation(dm_layers, dv_layers)
            logger.info(f'ROUND {round} DoRA Server Corr (layer-wise): {round_corr:.4f}')
            is_checkpoint = (round + 1) % checkpoint_interval == 0 or round == args.round - 1
            if is_checkpoint:
                dora_server_dm_accum.append(dm_layers)
                dora_server_dv_accum.append(dv_layers)
            
            # Calculate metrics for each client vs server
            l2_m_list, l2_v_list = [], []
            kl_m_list, kl_v_list = [], []
            
            for cid, client_w in updated_client_weights.items():
                local_model.load_state_dict(client_w)
                client_dora = get_dora_components(local_model)
                
                cid_l2_m, cid_l2_v = 0.0, 0.0
                cid_kl_m, cid_kl_v = 0.0, 0.0
                count = 0
                
                for name in server_dora.keys():
                    if name in client_dora:
                        m_c, V_c = client_dora[name]['m'], client_dora[name]['V']
                        m_s, V_s = server_dora[name]['m'], server_dora[name]['V']
                        
                        # L2
                        l2_m, l2_v = sim_calc.compute_dora_divergence(m_c, m_s, V_c, V_s, method='l2', mode='filter')
                        # KL
                        kl_m, kl_v = sim_calc.compute_dora_divergence(m_c, m_s, V_c, V_s, method='kl', mode='filter')
                        
                        cid_l2_m += l2_m.item()
                        cid_l2_v += l2_v.item()
                        cid_kl_m += kl_m.item()
                        cid_kl_v += kl_v.item()
                        count += 1
                        
                if count > 0:
                    l2_m_list.append(cid_l2_m / count)
                    l2_v_list.append(cid_l2_v / count)
                    kl_m_list.append(cid_kl_m / count)
                    kl_v_list.append(cid_kl_v / count)
            
            if l2_m_list:
                avg_l2_m = sum(l2_m_list) / len(l2_m_list)
                avg_l2_v = sum(l2_v_list) / len(l2_v_list)
                avg_kl_m = sum(kl_m_list) / len(kl_m_list)
                avg_kl_v = sum(kl_v_list) / len(kl_v_list)
                
                logger.info(f'ROUND {round} Client-Server Similarity -> '
                            f'L2(m): {avg_l2_m:.4f}, L2(V): {avg_l2_v:.4f} | '
                            f'KL(m): {avg_kl_m:.4f}, KL(V): {avg_kl_v:.4f}')

        test_acc = compute_accuracy(global_model, global_test_dataloader, device=device)
        pkl_dict['acc'].append(test_acc)
        logger.info(f'>> Global Model Test accuracy: {test_acc:.4f}')

    if args.peft == 'dora' and len(dora_server_dm_accum) >= 2:
        import numpy as np
        dm_all = np.concatenate(dora_server_dm_accum)
        dv_all = np.concatenate(dora_server_dv_accum)
        final_corr = compute_temporal_dora_correlation(dm_all, dv_all)
        pkl_dict['dora_final_corr'] = final_corr
        logger.info(f'DoRA Server Final Correlation (layer×checkpoint): {final_corr:.4f}')

    with open(os.path.join(args.logdir, log_file_name + '.pkl'), 'wb') as f:
        pickle.dump(pkl_dict, f)



if __name__ == '__main__':
    main()
