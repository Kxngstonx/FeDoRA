import torch
import torch.optim as optim
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import time
import copy
import os

from algs.feddecorr.decorr import FedDecorrLoss
from algs.fedproc.NegLoss import NegLoss_new
from algs.fedproc.PosLoss import PosLoss_new
from algs.fedsw.SWLoss import SampleWiseLoss
from algs.fedrcl.RCLLoss import RCLloss
from algs.fedsea.SEALoss import *


# Train module for Local Client


def compute_covariance_matrices(embeddings, labels):
    """
    embeddings: tensor of shape (N, D)
    labels: tensor of shape (N,)

    Returns:
      S_W: within-class covariance matrix, shape (D, D)
      S_B: between-class covariance matrix, shape (D, D)
    """
    overall_mean = embeddings.mean(dim=0, keepdim=True)
    unique_labels = torch.unique(labels)
    D = embeddings.shape[1]

    S_W = torch.zeros((D, D), device=embeddings.device)
    S_B = torch.zeros((D, D), device=embeddings.device)

    for cls in unique_labels:
        cls_mask = labels == cls
        cls_emb = embeddings[cls_mask]  # (n_c, D)
        cls_mean = cls_emb.mean(dim=0, keepdim=True)  # (1, D)
        S_W += (cls_emb - cls_mean).T @ (cls_emb - cls_mean)

        n_cls = cls_emb.shape[0]
        diff = cls_mean - overall_mean  # (1, D)
        S_B += n_cls * (diff.T @ diff)

    return S_W, S_B


def fedavg(net, train_dataloader, optimizer, device, args):
    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()

    total_loss = 0.0

    all_features = []
    all_labels = []
    total_batches = 0
    net.train()

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            total_batches += 1

            # --- DoRA Magnitude Warmup Logic ---
            args._is_warmup_step = False
            if getattr(args, 'peft', 'none') == 'dora' and getattr(args, 'dora_warmup_ratio', 0.0) > 0.0:
                total_expected_steps = len(train_dataloader) * args.epochs
                warmup_steps = int(total_expected_steps * args.dora_warmup_ratio)
                current_global_step = epoch * len(train_dataloader) + step

                if current_global_step < warmup_steps:
                    args._is_warmup_step = True
                    # Freeze A and B, Unfreeze m
                    for name, param in net.named_parameters():
                        if 'lora_B' in name:
                            param.requires_grad = False
                        elif name.endswith('.m'):
                            param.requires_grad = True
                elif current_global_step == warmup_steps:
                    # Unfreeze A and B
                    for name, param in net.named_parameters():
                        if 'lora_B' in name or name.endswith('.m'):
                            param.requires_grad = True
            # -----------------------------------

            if isinstance(x, dict):
                x = {k: v.to(device) for k, v in x.items()}
                target = target.to(device)
            else:
                x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()

            features, out = net(x)
            loss = criterion(out.float(), target)
            total_loss += loss.item()

            if args.feddecorr:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            loss.backward()
            
            # Boost gradient for m during warmup
            if getattr(args, 'peft', 'none') == 'dora' and getattr(args, 'dora_warmup_ratio', 0.0) > 0.0:
                if getattr(args, '_is_warmup_step', False):
                    multiplier = getattr(args, 'dora_warmup_lr_mult', 1.0)
                    if multiplier != 1.0:
                        for name, param in net.named_parameters():
                            if name.endswith('.m') and param.grad is not None:
                                param.grad *= multiplier
                                
            optimizer.step()


            all_features.append(features.detach())
            all_labels.append(target.detach())

    if total_batches > 0 and len(all_features) > 0:
        all_features = torch.cat(all_features, dim=0)  # shape: (N_total, D)
        all_labels = torch.cat(all_labels, dim=0)        # shape: (N_total,)
        S_W, S_B = compute_covariance_matrices(all_features, all_labels)
        avg_loss = total_loss / total_batches
    else:
        # no training batches for this client
        device_used = device if isinstance(
            device, torch.device) else torch.device(device)
        S_W = torch.zeros((1, 1), device=device_used)
        S_B = torch.zeros((1, 1), device=device_used)
        avg_loss = 0.0

    net.zero_grad()
    return avg_loss, S_W, S_B


def feddecorr(net, train_dataloader, optimizer, device, args):
    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()

    total_loss = 0.0

    all_features = []
    all_labels = []
    total_batches = 0
    net.train()

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            total_batches += 1
            
            # --- DoRA Magnitude Warmup Logic ---
            args._is_warmup_step = False
            if getattr(args, 'peft', 'none') == 'dora' and getattr(args, 'dora_warmup_ratio', 0.0) > 0.0:
                total_expected_steps = len(train_dataloader) * args.epochs
                warmup_steps = int(total_expected_steps * args.dora_warmup_ratio)
                current_global_step = epoch * len(train_dataloader) + step
                
                if current_global_step < warmup_steps:
                    args._is_warmup_step = True
                    # Freeze A and B, Unfreeze m
                    for name, param in net.named_parameters():
                        if 'lora_B' in name:
                            param.requires_grad = False
                        elif name.endswith('.m'):
                            param.requires_grad = True
                elif current_global_step == warmup_steps:
                    # Unfreeze A and B
                    for name, param in net.named_parameters():
                        if 'lora_B' in name or name.endswith('.m'):
                            param.requires_grad = True
            # -----------------------------------
            
            if isinstance(x, dict):
                x = {k: v.to(device) for k, v in x.items()}
                target = target.to(device)
            else:
                x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()

            features, out = net(x)
            loss = criterion(out.float(), target)
            total_loss += loss.item()

            if args.feddecorr:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            
            loss.backward()
            
            # Boost gradient for m during warmup
            if getattr(args, 'peft', 'none') == 'dora' and getattr(args, 'dora_warmup_ratio', 0.0) > 0.0:
                if getattr(args, '_is_warmup_step', False):
                    multiplier = getattr(args, 'dora_warmup_lr_mult', 1.0)
                    if multiplier != 1.0:
                        for name, param in net.named_parameters():
                            if name.endswith('.m') and param.grad is not None:
                                param.grad *= multiplier
                                
            optimizer.step()


            all_features.append(features.detach())
            all_labels.append(target.detach())

    if total_batches > 0 and len(all_features) > 0:
        all_features = torch.cat(all_features, dim=0)  # shape: (N_total, D)
        all_labels = torch.cat(all_labels, dim=0)        # shape: (N_total,)
        S_W, S_B = compute_covariance_matrices(all_features, all_labels)
        avg_loss = total_loss / total_batches
    else:
        device_used = device if isinstance(
            device, torch.device) else torch.device(device)
        S_W = torch.zeros((1, 1), device=device_used)
        S_B = torch.zeros((1, 1), device=device_used)
        avg_loss = 0.0

    net.zero_grad()
    return avg_loss, S_W, S_B


def fedprox(net, global_model, train_dataloader, optimizer, device, args):
    total_loss = 0.

    all_features = []
    all_labels = []
    net.train()

    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()
    global_weight_collector = list(global_model.parameters())

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            if isinstance(x, dict):
                x = {k: v.to(device) for k, v in x.items()}
                target = target.to(device)
            else:
                x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()
            features, out = net(x)
            loss = criterion(out.float(), target)

            if args.feddecorr:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((args.mu / 2) * torch.norm((param -
                                 global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg
            total_loss += loss.item()

            
            loss.backward()
            
            # Boost gradient for m during warmup
            if getattr(args, 'peft', 'none') == 'dora' and getattr(args, 'dora_warmup_ratio', 0.0) > 0.0:
                if getattr(args, '_is_warmup_step', False):
                    multiplier = getattr(args, 'dora_warmup_lr_mult', 1.0)
                    if multiplier != 1.0:
                        for name, param in net.named_parameters():
                            if name.endswith('.m') and param.grad is not None:
                                param.grad *= multiplier
                                
            optimizer.step()


            all_features.append(features.detach())
            all_labels.append(target.detach())

    all_features = torch.cat(all_features, dim=0)  # shape: (N_total, D)
    all_labels = torch.cat(all_labels, dim=0)        # shape: (N_total,)

    S_W, S_B = compute_covariance_matrices(all_features, all_labels)

    net.zero_grad()
    return total_loss / len(train_dataloader) / args.epochs, S_W, S_B


def moon(net, global_model, previous_net, train_dataloader, optimizer, device, args):
    total_loss = 0.

    all_features = []
    all_labels = []
    net.train()

    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()
    cos = torch.nn.CosineSimilarity(dim=-1)

    for epoch in range(args.epochs):

        for step, (x, target) in enumerate(train_dataloader):
            if isinstance(x, dict):
                x = {k: v.to(device) for k, v in x.items()}
                target = target.to(device)
            else:
                x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()
            features1, out = net(x)
            features2, _ = global_model(x)

            posi = cos(features1, features2)
            logits = posi.reshape(-1, 1)

            features3, _ = previous_net(x)
            nega = cos(features1, features3)
            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

            logits /= args.temperature
            labels = torch.zeros(x.size(0)).long().to(device)
            loss2 = args.mu * criterion(logits, labels)
            loss1 = criterion(out.float(), target)
            loss = loss1 + loss2
            total_loss += loss.item()

            if args.feddecorr:
                loss_feddecorr = feddecorr(features1)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            
            loss.backward()
            
            # Boost gradient for m during warmup
            if getattr(args, 'peft', 'none') == 'dora' and getattr(args, 'dora_warmup_ratio', 0.0) > 0.0:
                if getattr(args, '_is_warmup_step', False):
                    multiplier = getattr(args, 'dora_warmup_lr_mult', 1.0)
                    if multiplier != 1.0:
                        for name, param in net.named_parameters():
                            if name.endswith('.m') and param.grad is not None:
                                param.grad *= multiplier
                                
            optimizer.step()


            all_features.append(features1.detach())
            all_labels.append(target.detach())

    all_features = torch.cat(all_features, dim=0)  # shape: (N_total, D)
    all_labels = torch.cat(all_labels, dim=0)        # shape: (N_total,)

    S_W, S_B = compute_covariance_matrices(all_features, all_labels)

    net.zero_grad()
    return total_loss / len(train_dataloader) / args.epochs, S_W, S_B


def fedproc(net, train_dataloader, optimizer, device, args, round, global_class_center):
    total_loss = 0.

    all_features = []
    all_labels = []
    net.train()

    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()
    pos_loss_fn = PosLoss_new()
    neg_loss_fn = NegLoss_new()

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            if isinstance(x, dict):
                x = {k: v.to(device) for k, v in x.items()}
                target = target.to(device)
            else:
                x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()
            features, out = net(x)
            CEloss = criterion(out.float(), target)

            if round < 1:
                loss = CEloss

            else:
                pos_logits = pos_loss_fn(
                    features, labels=target, center=global_class_center, args=args)
                neg_logits = neg_loss_fn(
                    features, labels=target, center=global_class_center, args=args)

                pos_sim = pos_logits[:, 0]
                neg_sim = neg_logits[:, 0]

                pos_term = -torch.mean(pos_sim)
                neg_term = torch.mean(neg_sim)

                SCloss = args.lambda_pull * pos_term + args.lambda_push * neg_term

                # alpha = (1 - round/args.round)
                # loss = alpha * CEloss + (1 - alpha) * SCloss
                loss = CEloss + SCloss

            total_loss += loss.item()

            if args.feddecorr:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            
            loss.backward()
            
            # Boost gradient for m during warmup
            if getattr(args, 'peft', 'none') == 'dora' and getattr(args, 'dora_warmup_ratio', 0.0) > 0.0:
                if getattr(args, '_is_warmup_step', False):
                    multiplier = getattr(args, 'dora_warmup_lr_mult', 1.0)
                    if multiplier != 1.0:
                        for name, param in net.named_parameters():
                            if name.endswith('.m') and param.grad is not None:
                                param.grad *= multiplier
                                
            optimizer.step()


            all_features.append(features.detach())
            all_labels.append(target.detach())

    all_features = torch.cat(all_features, dim=0)  # shape: (N_total, D)
    all_labels = torch.cat(all_labels, dim=0)        # shape: (N_total,)

    S_W, S_B = compute_covariance_matrices(all_features, all_labels)

    net.zero_grad()
    return total_loss / len(train_dataloader) / args.epochs, S_W, S_B


def fedsw(net, train_dataloader, optimizer, device, args):
    total_loss = 0.

    all_features = []
    all_labels = []
    net.train()

    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()
    criterion_sw = SampleWiseLoss()

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            if isinstance(x, dict):
                x = {k: v.to(device) for k, v in x.items()}
                target = target.to(device)
            else:
                x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()

            features, out = net(x)
            CEloss = criterion(out.float(), target)

            features_norm = F.normalize(features, p=2, dim=1)
            cosine_sim_matrix = torch.mm(features_norm, features_norm.t())

            sw_loss = criterion_sw(
                cosine_sim_matrix=cosine_sim_matrix, x=x, target=target, args=args)
            loss = CEloss + (args.mu * sw_loss)

            total_loss += loss.item()

            if args.feddecorr:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            
            loss.backward()
            
            # Boost gradient for m during warmup
            if getattr(args, 'peft', 'none') == 'dora' and getattr(args, 'dora_warmup_ratio', 0.0) > 0.0:
                if getattr(args, '_is_warmup_step', False):
                    multiplier = getattr(args, 'dora_warmup_lr_mult', 1.0)
                    if multiplier != 1.0:
                        for name, param in net.named_parameters():
                            if name.endswith('.m') and param.grad is not None:
                                param.grad *= multiplier
                                
            optimizer.step()


            all_features.append(features.detach())
            all_labels.append(target.detach())

    all_features = torch.cat(all_features, dim=0)  # shape: (N_total, D)
    all_labels = torch.cat(all_labels, dim=0)        # shape: (N_total,)

    S_W, S_B = compute_covariance_matrices(all_features, all_labels)

    net.zero_grad()
    return total_loss / len(train_dataloader) / args.epochs, S_W, S_B


def fedrcl(net, train_dataloader, optimizer, device, args):
    total_loss = 0.

    all_features = []
    all_labels = []
    net.train()

    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()
    criterion_rcl = RCLloss()

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            if isinstance(x, dict):
                x = {k: v.to(device) for k, v in x.items()}
                target = target.to(device)
            else:
                x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()

            features, out = net(x)
            CEloss = criterion(out.float(), target)

            features_norm = F.normalize(features, p=2, dim=1)
            cosine_sim_matrix = torch.mm(features_norm, features_norm.t())

            rcl_loss = criterion_rcl(
                cosine_sim_matrix=cosine_sim_matrix, x=x, target=target, args=args)
            loss = CEloss + rcl_loss

            total_loss += loss.item()

            if args.feddecorr:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            
            loss.backward()
            
            # Boost gradient for m during warmup
            if getattr(args, 'peft', 'none') == 'dora' and getattr(args, 'dora_warmup_ratio', 0.0) > 0.0:
                if getattr(args, '_is_warmup_step', False):
                    multiplier = getattr(args, 'dora_warmup_lr_mult', 1.0)
                    if multiplier != 1.0:
                        for name, param in net.named_parameters():
                            if name.endswith('.m') and param.grad is not None:
                                param.grad *= multiplier
                                
            optimizer.step()


            all_features.append(features.detach())
            all_labels.append(target.detach())

    all_features = torch.cat(all_features, dim=0)  # shape: (N_total, D)
    all_labels = torch.cat(all_labels, dim=0)        # shape: (N_total,)

    S_W, S_B = compute_covariance_matrices(all_features, all_labels)

    net.zero_grad()
    return total_loss / len(train_dataloader) / args.epochs, S_W, S_B


def decfedproc(net, train_dataloader, optimizer, device, args, round, global_class_center):
    total_loss = 0.

    all_features = []
    all_labels = []
    net.train()

    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()
    pos_loss_fn = PosLoss_new()
    neg_loss_fn = NegLoss_new()

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            if isinstance(x, dict):
                x = {k: v.to(device) for k, v in x.items()}
                target = target.to(device)
            else:
                x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()
            features, out = net(x)
            CEloss = criterion(out.float(), target)

            if round < 1:
                loss = CEloss

            else:
                pos_logits = pos_loss_fn(
                    features, labels=target, center=global_class_center, args=args)
                neg_logits = neg_loss_fn(
                    features, labels=target, center=global_class_center, args=args)

                pos_sim = pos_logits[:, 0]
                neg_sim = neg_logits[:, 0]

                pos_term = -torch.mean(pos_sim)
                neg_term = torch.mean(neg_sim)

                SCloss = args.lambda_pull * pos_term + args.lambda_push * neg_term

                loss = CEloss + args.mu * SCloss

            total_loss += loss.item()

            if args.feddecorr:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            
            loss.backward()
            
            # Boost gradient for m during warmup
            if getattr(args, 'peft', 'none') == 'dora' and getattr(args, 'dora_warmup_ratio', 0.0) > 0.0:
                if getattr(args, '_is_warmup_step', False):
                    multiplier = getattr(args, 'dora_warmup_lr_mult', 1.0)
                    if multiplier != 1.0:
                        for name, param in net.named_parameters():
                            if name.endswith('.m') and param.grad is not None:
                                param.grad *= multiplier
                                
            optimizer.step()


            all_features.append(features.detach())
            all_labels.append(target.detach())

    all_features = torch.cat(all_features, dim=0)  # shape: (N_total, D)
    all_labels = torch.cat(all_labels, dim=0)      # shape: (N_total,)

    S_W, S_B = compute_covariance_matrices(all_features, all_labels)

    net.zero_grad()
    return total_loss / len(train_dataloader) / args.epochs, S_W, S_B


def decfedsw(net, train_dataloader, optimizer, device, args):
    total_loss = 0.

    all_features = []
    all_labels = []
    net.train()

    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()
    criterion_sw = SampleWiseLoss()

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            if isinstance(x, dict):
                x = {k: v.to(device) for k, v in x.items()}
                target = target.to(device)
            else:
                x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()

            features, out = net(x)
            CEloss = criterion(out.float(), target)

            features_norm = F.normalize(features, p=2, dim=1)
            cosine_sim_matrix = torch.mm(features_norm, features_norm.t())

            sw_loss = criterion_sw(
                cosine_sim_matrix=cosine_sim_matrix, x=x, target=target, args=args)
            loss = CEloss + (args.mu * sw_loss)

            total_loss += loss.item()

            if args.feddecorr:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            
            loss.backward()
            
            # Boost gradient for m during warmup
            if getattr(args, 'peft', 'none') == 'dora' and getattr(args, 'dora_warmup_ratio', 0.0) > 0.0:
                if getattr(args, '_is_warmup_step', False):
                    multiplier = getattr(args, 'dora_warmup_lr_mult', 1.0)
                    if multiplier != 1.0:
                        for name, param in net.named_parameters():
                            if name.endswith('.m') and param.grad is not None:
                                param.grad *= multiplier
                                
            optimizer.step()


            all_features.append(features.detach())
            all_labels.append(target.detach())

    all_features = torch.cat(all_features, dim=0)  # shape: (N_total, D)
    all_labels = torch.cat(all_labels, dim=0)        # shape: (N_total,)

    S_W, S_B = compute_covariance_matrices(all_features, all_labels)

    net.zero_grad()
    return total_loss / len(train_dataloader) / args.epochs, S_W, S_B


def fedsol(net, global_model, train_dataloader, optimizer, device, args):
    total_loss = 0.
    all_features = []
    all_labels = []

    net.train()
    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()
    KLDiv = nn.KLDivLoss(reduction="batchmean")
    perturb_head = True
    perturb_body = True

    dg_model = copy.deepcopy(global_model)
    dg_model.to(device)
    for params in dg_model.parameters():
        params.requires_grad = False

    if args.adaptive:
        sam_optimizer = get_sam_optimizer_adaptive(
            net, dg_model, optimizer, args)
    else:
        sam_optimizer = get_sam_optimizer_fixed(net, optimizer, args)

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            if isinstance(x, dict):
                x = {k: v.to(device) for k, v in x.items()}
                target = target.to(device)
            else:
                x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()

            enable_running_stats(net)

            if not perturb_head:
                freeze_head(net)

            if not perturb_body:
                freeze_body(net)

            _, logits = net(x)
            _, dg_logits = dg_model(x)

            with torch.no_grad():
                dg_probs = torch.softmax(dg_logits / 3, dim=1)
            pred_probs = F.log_softmax(logits / 3, dim=1)

            loss = KLDiv(pred_probs, dg_probs)
            grads = torch.autograd.grad(
                loss,
                net.parameters(),
                create_graph=True
            )
            for p, g in zip(net.parameters(), grads):
                p.grad = g

            if not perturb_head:
                zerograd_head(net)

            if not perturb_body:
                zerograd_body(net)

            sam_optimizer.first_step(zero_grad=True)

            unfreeze(net)

            # second forward-backward pass
            disable_running_stats(net)
            features, logits_perturbed = net(x)
            sam_loss = criterion(logits_perturbed, target)
            sam_loss.backward()  # make sure to do a full forward pass
            sam_optimizer.second_step(zero_grad=True)

            all_features.append(features.detach())
            all_labels.append(target.detach())

            total_loss += (loss.item() + sam_loss.item())

    all_features = torch.cat(all_features, dim=0)  # shape: (N_total, D)
    all_labels = torch.cat(all_labels, dim=0)        # shape: (N_total,)

    S_W, S_B = compute_covariance_matrices(all_features, all_labels)

    net.zero_grad()
    return total_loss / len(train_dataloader) / args.epochs, S_W, S_B


def fedsea(net, iid_generator, discriminator, dataloader, optimizer_net, optimizer_gen,
           device, args, mask, client_id, lambda_fedsea):
    all_features = []
    all_labels = []

    net.train()
    iid_generator.train()
    discriminator.eval()

    criterion_task = nn.CrossEntropyLoss()
    criterion_discriminator = nn.CrossEntropyLoss()

    total_task_loss = 0.0
    total_gen_loss = 0.0

    client_labels = torch.full(
        (args.batch_size,), client_id, dtype=torch.long, device=device)

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(dataloader):
            if x.size(0) != args.batch_size:
                client_labels = torch.full(
                    (x.size(0),), client_id, dtype=torch.long, device=device)

            if isinstance(x, dict):
                x = {k: v.to(device) for k, v in x.items()}
                target = target.to(device)
            else:
                x, target = x.to(device), target.to(device).long()

            features, out = net(x)

            loss_task = criterion_task(out, target)

            features_for_generator = features.detach().clone()

            aligned_features = iid_generator(features_for_generator, mask)
            reversed_aligned_features = grad_reverse(
                aligned_features, lambda_fedsea)
            client_logits_for_gen = discriminator(reversed_aligned_features)
            loss_discriminator_for_gen = criterion_discriminator(
                client_logits_for_gen, client_labels)

            optimizer_net.zero_grad()
            loss_task.backward()
            optimizer_net.step()

            optimizer_gen.zero_grad()
            loss_discriminator_for_gen.backward()
            optimizer_gen.step()

            total_task_loss += loss_task.item()
            # total_gen_loss += loss_discriminator_for_gen.item()

            all_features.append(features.detach())
            all_labels.append(target.detach())

    avg_task_loss = total_task_loss / (len(dataloader) * args.epochs)

    all_features = torch.cat(all_features, dim=0)  # shape: (N_total, D)
    all_labels = torch.cat(all_labels, dim=0)        # shape: (N_total,)

    S_W, S_B = compute_covariance_matrices(all_features, all_labels)

    net.zero_grad()
    iid_generator.zero_grad()
    return avg_task_loss, S_W, S_B


def adjust_lr(round, current_lr, args):
    if args.scheduler == 'linear':
        new_lr = args.eta_min + (args.lr - args.eta_min) * \
            (1 - round / args.round)
    elif args.scheduler == 'cosine':
        new_lr = args.eta_min + (args.lr - args.eta_min) * \
            0.5 * (1 + math.cos(math.pi * round / args.round))
    elif args.scheduler == 'step':
        # Calculate decay purely from the initial args.lr to prevent state mutation bugs
        decay_steps = 0
        for r in args.schedule_round:
            if round + 1 >= r:
                decay_steps += 1
        new_lr = args.lr * (args.lr_gamma ** decay_steps)
    else:
        new_lr = args.lr
    return new_lr


@torch.no_grad()
def _get_dg_logits(data, dg_model):
    dg_logits = dg_model(data)

    return dg_logits


def train_local_net(dataloaders, nets, global_model, prev_nets, device, round, lr, args, logger, global_class_center, iid_generators=None, discriminator=None, mask=None):
    total_loss = 0.0
    lr = adjust_lr(round, lr, args)
    list_S_W = []
    list_S_B = []
    for net_id, net in nets.items():
        net.train()
        
        # Optimizer group setup for DoRA differential LR / WD
        params_group = []
        if getattr(args, 'peft', 'none') == 'dora' and (getattr(args, 'dora_m_lr', None) is not None or getattr(args, 'dora_m_wd', None) is not None):
            m_params = []
            other_params = []
            for name, param in net.named_parameters():
                if not param.requires_grad:
                    continue
                if name.endswith('.m') or '.m.' in name:
                    m_params.append(param)
                else:
                    other_params.append(param)
            
            # Use main LR and reg as defaults if not overridden
            dora_m_lr = args.dora_m_lr if getattr(args, 'dora_m_lr', None) is not None else lr
            dora_m_wd = args.dora_m_wd if getattr(args, 'dora_m_wd', None) is not None else args.reg
            
            params_group = [
                {'params': other_params},
                {'params': m_params, 'lr': dora_m_lr, 'weight_decay': dora_m_wd}
            ]
        else:
            params_group = [{'params': filter(lambda p: p.requires_grad, net.parameters())}]

        if args.optimizer == 'adam':
            optimizer = optim.Adam(params_group,
                                   lr=lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(params_group,
                                   lr=lr, weight_decay=args.reg, amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(params_group,
                                  lr=lr, momentum=args.momentum, weight_decay=args.reg)
            if args.alg == 'fedsea':
                optimizer_gen = optim.SGD(filter(lambda p: p.requires_grad, iid_generators[net_id].parameters(
                )), lr=lr, momentum=args.momentum, weight_decay=args.reg)

        if args.alg == 'fedavg':
            loss, S_W, S_B = fedavg(
                net, dataloaders[net_id], optimizer, device, args)
            list_S_W.append(S_W)
            list_S_B.append(S_B)

            total_loss += loss

        if args.alg == 'feddecorr':
            loss, S_W, S_B = feddecorr(
                net, dataloaders[net_id], optimizer, device, args)
            list_S_W.append(S_W)
            list_S_B.append(S_B)

            total_loss += loss

        elif args.alg == 'fedprox':
            loss, S_W, S_B = fedprox(
                net, global_model, dataloaders[net_id], optimizer, device, args)
            list_S_W.append(S_W)
            list_S_B.append(S_B)

            total_loss += loss

        elif args.alg == 'moon':
            loss, S_W, S_B = moon(
                net, global_model, prev_nets[net_id], dataloaders[net_id], optimizer, device, args)
            list_S_W.append(S_W)
            list_S_B.append(S_B)

            total_loss += loss

        elif args.alg == 'fedproc':
            loss, S_W, S_B = fedproc(
                net, dataloaders[net_id], optimizer, device, args, round, global_class_center)
            list_S_W.append(S_W)
            list_S_B.append(S_B)

            total_loss += loss

        elif args.alg == 'decfedproc':
            loss, S_W, S_B = decfedproc(
                net, dataloaders[net_id], optimizer, device, args, round, global_class_center)
            list_S_W.append(S_W)
            list_S_B.append(S_B)

            total_loss += loss

        elif args.alg == 'fedsw':
            loss, S_W, S_B = fedsw(
                net, dataloaders[net_id], optimizer, device, args)
            list_S_W.append(S_W)
            list_S_B.append(S_B)

            total_loss += loss

        elif args.alg == 'decfedsw':
            loss, S_W, S_B = decfedsw(
                net, dataloaders[net_id], optimizer, device, args)
            list_S_W.append(S_W)
            list_S_B.append(S_B)

            total_loss += loss

        elif args.alg == 'fedrcl':
            loss, S_W, S_B = fedrcl(
                net, dataloaders[net_id], optimizer, device, args)
            list_S_W.append(S_W)
            list_S_B.append(S_B)

            total_loss += loss

        elif args.alg == 'fedsea':
            current_mask = mask.to(device) if mask is not None else None
            discriminator.to(device)
            lambda_fedsea = round / (args.round-1)

            loss, S_W, S_B = fedsea(
                net=net,
                iid_generator=iid_generators[net_id],
                discriminator=discriminator,
                dataloader=dataloaders[net_id],
                optimizer_net=optimizer,
                optimizer_gen=optimizer_gen,
                device=device,
                args=args,
                mask=current_mask,
                client_id=net_id,
                lambda_fedsea=lambda_fedsea
            )
            list_S_W.append(S_W)
            list_S_B.append(S_B)

            total_loss += loss
        elif args.alg == 'fedbabu':
            # FedBABU: same local update as FedAvg, but classifier/head is expected to be frozen externally
            loss, S_W, S_B = fedavg(
                net, dataloaders[net_id], optimizer, device, args)
            list_S_W.append(S_W)
            list_S_B.append(S_B)
            total_loss += loss

    # Handle case where no S_W/S_B were collected to avoid stack() runtime error
    if len(list_S_W) > 0:
        avg_S_W = torch.stack(list_S_W, dim=0).mean(dim=0)
        avg_S_B = torch.stack(list_S_B, dim=0).mean(dim=0)
    else:
        # fallback: set 1x1 zero tensors to allow tracing/logging without errors
        avg_S_W = torch.zeros((1, 1), device=device)
        avg_S_B = torch.zeros((1, 1), device=device)

    within_variance_scalar = torch.trace(avg_S_W)
    between_variance_scalar = torch.trace(avg_S_B)

    avg_loss = total_loss / len(nets) if len(nets) > 0 else 0.0
    logger.info(
        f'At round: {round}, avg_loss: {avg_loss:.4f}, within_variance: {(within_variance_scalar.item())}, between_variance: {between_variance_scalar.item()}')

    return avg_loss, lr


def fedsea_server_train(global_model, fedsea_dataloaders, optimizer_discriminator, optimizer_attention, client_discriminator, client_iid_generators, attention_p, feature_dim, clients_this_round, args):
    client_discriminator.train()
    all_features_list = []
    all_client_ids_list = []

    global_model.eval()
    t1 = time.time()
    with torch.no_grad():
        for client_id in range(args.n_clients):
            all_images_tensor, all_labels_tensor = next(
                iter(fedsea_dataloaders[client_id]))
            features, _ = global_model(all_images_tensor.to(args.device))
            all_features_list.append(features)
            all_client_ids_list.append(torch.full(
                (features.size(0),), client_id, dtype=torch.long))

    all_features = torch.cat(all_features_list, dim=0).to(args.device)
    all_client_ids = torch.cat(all_client_ids_list, dim=0).to(args.device)
    t2 = time.time()
    print(f'FedSea feature collection: {t2-t1}')

    criterion_discriminator_server = nn.CrossEntropyLoss()
    for server_epoch in range(args.server_epochs_fedsea):
        permutation = torch.randperm(all_features.size(0))
        all_features_shuffled = all_features[permutation]
        all_client_ids_shuffled = all_client_ids[permutation]

        for i in range(0, all_features.size(0), args.server_batch_size_fedsea):
            batch_features = all_features_shuffled[i: i +
                                                   args.server_batch_size_fedsea]
            batch_labels = all_client_ids_shuffled[i: i +
                                                   args.server_batch_size_fedsea]

            if batch_features.size(0) == 0:
                continue

            optimizer_discriminator.zero_grad()
            optimizer_attention.zero_grad()

            current_attention_vector = torch.sigmoid(
                args.attn_scale_fedsea * attention_p)

            client_logits = client_discriminator(
                batch_features, current_attention_vector)

            loss_d = criterion_discriminator_server(
                client_logits, batch_labels)

            loss_d.backward()
            optimizer_discriminator.step()
            optimizer_attention.step()

    print(f"FedSea: Server update complete. Last loss_d: {loss_d.item():.4f}")

    final_attention_vector = torch.sigmoid(
        args.attn_scale_fedsea * attention_p).detach()
    feature_mask = get_mask_from_attention(
        final_attention_vector, args.mask_k_fedsea)  # k from args
    print(
        f"FedSea: Generated new mask M. Features to align: {int(feature_mask.sum())}/{feature_dim}")

    iid_generators_this_round = {
        i: client_iid_generators[i] for i in clients_this_round}
    for gen in iid_generators_this_round.values():
        gen.to(args.device)

    return feature_mask, iid_generators_this_round
