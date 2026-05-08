import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import json
import copy
import torch.nn.functional as F

import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from datasets.cifar10 import CIFAR10_truncated
from datasets.cifar100 import CIFAR100_truncated
from datasets.fmnist import FashionMNIST_truncated
from datasets.folder import ImageFolder_custom
from datasets.wrapper import AugmentedDatasetWrapper
from datasets.femnist import FEMNIST_truncated
from datasets.ag_news import AGNewsDataset

from models import resnet_cifar
from models import mobilenet
from models.qwen_llm import QwenLLMWrapper
from models.vit_vision import ViTWrapper

from algs.fedsea.SEALoss import *


def init_logger(args):
    os.makedirs(args.logdir, exist_ok=True)
    if getattr(args, 'freeze_layers_list', None) and len(args.freeze_layers_list) > 0:
        freeze_info = ','.join(map(str, args.freeze_layers_list))
    else:
        freeze_info = str(args.freeze_layer)
    log_file_name = f'{args.alg}_{args.seed}_{args.dataset}_partition{args.partition}_beta{args.beta}_frac{args.sample_fraction}_momentum{args.momentum}_pull{args.lambda_pull}_push{args.lambda_push}_frozen_layers{freeze_info}_frozen_classifier{args.freeze_classifier}_freeze_block_pos{args.freeze_block_pos}'
    if args.log_file_name is None:
        log_file_name = f'log_{log_file_name}'
    else:
        log_file_name = args.log_file_name
    with open(os.path.join(args.logdir, log_file_name + '.json'), 'w') as f:
        args_dict = vars(args)
        json.dump(args_dict, f, indent=4, ensure_ascii=False)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=os.path.join(args.logdir, log_file_name + '.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s %(message)s',
        datefmt='%m-%d %H:%M'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return logger, log_file_name


def get_hf_model_name(model_arg):
    mapping = {
        'qwen': 'Qwen/Qwen2.5-0.5B',
        'roberta-base': 'roberta-base',
        'roberta-large': 'roberta-large',
        'distilbert': 'distilbert-base-uncased',
        'llama-3.2-1b': 'meta-llama/Llama-3.2-1B'
    }
    return mapping.get(model_arg, model_arg)


def get_global_dataset(args):
    model_name = get_hf_model_name(getattr(args, 'model', 'roberta-base'))
    
    if args.dataset.startswith('glue_'):
        from datasets.glue import GLUEDataset
        task = args.dataset[len('glue_'):]
        train_ds = GLUEDataset(task=task, train=True,  model_name=model_name, max_length=128)
        val_ds   = None
        test_ds  = GLUEDataset(task=task, train=False, model_name=model_name, max_length=128)
        return train_ds, val_ds, test_ds

    if args.dataset == 'ag_news':
        train_ds = AGNewsDataset(
            train=True, model_name=model_name, max_length=128)
        val_ds = None
        test_ds = AGNewsDataset(
            train=False, model_name=model_name, max_length=128)
        return train_ds, val_ds, test_ds

    if args.dataset == 'banking77':
        from datasets.banking77 import Banking77Dataset
        train_ds = Banking77Dataset(
            train=True, model_name=model_name, max_length=128)
        val_ds = None
        test_ds = Banking77Dataset(
            train=False, model_name=model_name, max_length=128)
        return train_ds, val_ds, test_ds

    if args.dataset == '20newsgroups':
        from datasets.newsgroups20 import Newsgroups20Dataset
        train_ds = Newsgroups20Dataset(
            train=True, model_name=model_name, max_length=128)
        val_ds = None
        test_ds = Newsgroups20Dataset(
            train=False, model_name=model_name, max_length=128)
        return train_ds, val_ds, test_ds

    if args.dataset == 'fmnist':
        normalize = transforms.Normalize(mean=[0.2860], std=[0.3530])

        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        # test set data prep
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        train_ds = FashionMNIST_truncated(
            args.datadir, train=True, transform=transform_train, download=True)
        val_ds = None
        test_ds = FashionMNIST_truncated(
            args.datadir, train=False, transform=transform_test, download=True)

    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        train_ds = CIFAR10_truncated(
            args.datadir, train=True, transform=transform_train, download=True)
        val_ds = None
        test_ds = CIFAR10_truncated(
            args.datadir, train=False, transform=transform_test, download=True)

    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        train_ds = CIFAR100_truncated(
            args.datadir, train=True, transform=transform_train, download=True)
        val_ds = None
        test_ds = CIFAR100_truncated(
            args.datadir, train=False, transform=transform_test, download=True)

    elif args.dataset == 'tinyimagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2770, 0.2691, 0.2821)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2770, 0.2691, 0.2821)),
        ])

        train_ds = ImageFolder_custom(os.path.join(
            args.datadir, 'train'), transform=transform_train)
        val_ds = None
        test_ds = ImageFolder_custom(os.path.join(
            args.datadir, 'val'), transform=transform_test)

    return train_ds, val_ds, test_ds


def record_net_data_stats(y_train, net_dataidx_map, logger):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.debug('Data statistics: %s' % str(net_cls_counts))
    return


def partition_data(global_train_dataset, args, logger):
    if hasattr(global_train_dataset, 'data') and hasattr(global_train_dataset, 'targets'):
        y_train = np.array(global_train_dataset.targets)
    elif hasattr(global_train_dataset, 'target'):
        y_train = np.array(global_train_dataset.target)
    elif hasattr(global_train_dataset, 'labels'):
        y_train = np.array(global_train_dataset.labels)
    elif hasattr(global_train_dataset, 'samples'):
        y_train = np.array([s[1] for s in global_train_dataset.samples])
    else:
        raise AttributeError(
            "Error"
        )

    n_train = y_train.shape[0]

    if hasattr(global_train_dataset, 'num_classes'):
        K = global_train_dataset.num_classes
    elif hasattr(global_train_dataset, 'classes'):
        K = len(global_train_dataset.classes)
    else:
        K = len(np.unique(y_train))

    net_dataidx_map = {}

    if args.partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, args.n_clients)
        net_dataidx_map = {i: batch_idxs[i] for i in range(args.n_clients)}

    elif args.partition == "noniid":
        min_size = 0
        N = n_train
        while min_size < args.min_require_size:
            idx_batch = [[] for _ in range(args.n_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                props = np.random.dirichlet(
                    np.repeat(args.beta, args.n_clients))
                props = np.array([
                    p * (len(idx_j) < N / args.n_clients)
                    for p, idx_j in zip(props, idx_batch)
                ])
                props = props / props.sum()
                splits = (np.cumsum(props) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, splits))
                ]
            min_size = min(len(idx_j) for idx_j in idx_batch)
        for j in range(args.n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = np.array(idx_batch[j], dtype='int64')

    elif args.partition == "noniid_balanced":
        N = n_train
        num_per_client = N // args.n_clients
        assigned = set()
        for i in range(args.n_clients):
            weights = torch.zeros(N)
            proportions = np.random.dirichlet(np.repeat(args.beta, K))
            for k, p in enumerate(proportions):
                idx_k = np.where(y_train == k)[0]
                weights[idx_k] = p
            weights[list(assigned)] = 0.0
            chosen = torch.multinomial(
                weights, num_per_client, replacement=False).tolist()
            assigned.update(chosen)
            net_dataidx_map[i] = np.array(chosen, dtype='int64')

    else:
        raise ValueError(f"Unknown partition type: {args.partition}")

    record_net_data_stats(y_train, net_dataidx_map, logger)
    return net_dataidx_map


def shuffle_clients(args):
    n_party_per_round = int(args.n_clients * args.sample_fraction)
    party_list = [i for i in range(args.n_clients)]
    party_list_rounds = []
    if n_party_per_round != args.n_clients:
        for i in range(args.round):
            party_list_rounds.append(
                random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.round):
            party_list_rounds.append(party_list)
    return party_list_rounds


def get_client_datasets(global_train_dataset, client_data_map, args):
    client_datasets = {}
    for i in range(args.n_clients):
        client_datasets[i] = (data.Subset(
            global_train_dataset, client_data_map[i]))

    return client_datasets


def get_client_meta_datasets(client_datasets, args):
    client_meta_datasets = {}
    transform = []
    for i in range(args.n_clients):
        client_meta_datasets[i] = (AugmentedDatasetWrapper(
            client_datasets[i], transform=transform))

    return client_meta_datasets


def get_global_dataloader(global_train_dataset, global_val_dataset, global_test_dataset, args):
    global_train_dataloader = data.DataLoader(dataset=global_train_dataset, batch_size=args.batch_size,
                                              drop_last=False, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    global_val_dataloader = None
    if global_val_dataset is not None:
        global_val_dataloader = data.DataLoader(
            dataset=global_val_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    global_test_dataloader = data.DataLoader(
        dataset=global_test_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    return global_train_dataloader, global_val_dataloader, global_test_dataloader


def get_client_dataloaders(client_datasets, args):
    dataloaders = {}
    for i in range(args.n_clients):
        client_train_dataloader = data.DataLoader(
            dataset=client_datasets[i], batch_size=args.batch_size, drop_last=True, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        dataloaders[i] = client_train_dataloader

    return dataloaders


def get_client_test_dataloaders(client_data_map, global_train_dataset, global_test_dataset, args):
    """Create a test dataloader for each client by selecting test samples whose labels appear in the client's training split.
    If no matching test samples exist for a client, fallback to the full test set for that client.
    Returns a dict: client_id -> DataLoader
    """
    # test labels
    if hasattr(global_test_dataset, 'targets'):
        test_labels = np.array(global_test_dataset.targets)
    elif hasattr(global_test_dataset, 'target'):
        test_labels = np.array(global_test_dataset.target)
    elif hasattr(global_test_dataset, 'labels'):
        test_labels = np.array(global_test_dataset.labels)
    elif hasattr(global_test_dataset, 'samples'):
        test_labels = np.array([s[1] for s in global_test_dataset.samples])
    elif hasattr(global_test_dataset, 'imgs'):
        test_labels = np.array([s[1] for s in global_test_dataset.imgs])
    elif hasattr(global_test_dataset, 'data') and hasattr(global_test_dataset, 'target'):
        # custom dataset that stores data and target arrays (e.g., ImageFolder_custom)
        test_labels = np.array(global_test_dataset.target)
    else:
        raise AttributeError(
            f"global_test_dataset lacks label attribute (type: {type(global_test_dataset)})")

    label_to_indices = {}
    for idx, lbl in enumerate(test_labels):
        label_to_indices.setdefault(lbl, []).append(idx)

    # train labels (to know which classes each client has)
    if hasattr(global_train_dataset, 'targets'):
        train_labels = np.array(global_train_dataset.targets)
    elif hasattr(global_train_dataset, 'target'):
        train_labels = np.array(global_train_dataset.target)
    elif hasattr(global_train_dataset, 'labels'):
        train_labels = np.array(global_train_dataset.labels)
    elif hasattr(global_train_dataset, 'samples'):
        train_labels = np.array([s[1] for s in global_train_dataset.samples])
    elif hasattr(global_train_dataset, 'imgs'):
        train_labels = np.array([s[1] for s in global_train_dataset.imgs])
    elif hasattr(global_train_dataset, 'data') and hasattr(global_train_dataset, 'target'):
        train_labels = np.array(global_train_dataset.target)
    else:
        raise AttributeError(
            f"global_train_dataset lacks label attribute (type: {type(global_train_dataset)})")

    client_test_loaders = {}
    for i in range(args.n_clients):
        train_idx = client_data_map.get(i, np.array([], dtype='int64'))
        if train_idx is None or len(train_idx) == 0:
            # fallback to full test set
            subset_indices = np.arange(len(test_labels))
        else:
            client_classes = set(train_labels[train_idx])
            subset_indices = []
            for cls in client_classes:
                subset_indices.extend(label_to_indices.get(cls, []))
            if len(subset_indices) == 0:
                subset_indices = np.arange(len(test_labels))

        subset = data.Subset(global_test_dataset, subset_indices)
        client_test_loaders[i] = data.DataLoader(
            dataset=subset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    return client_test_loaders


def prepare_data_for_training(args, logger):
    """Prepare global datasets, dataloaders, and client-level datasets/dataloaders.

    Returns a tuple of:
      (global_train_dataset, global_val_dataset, global_test_dataset,
       client_data_map, clients_at_rounds, client_datasets,
       global_train_dataloader, global_val_dataloader, global_test_dataloader,
       client_dataloaders, client_test_dataloaders, client_finetune_train)
    """
    global_train_dataset, global_val_dataset, global_test_dataset = get_global_dataset(
        args)

    # dataidx_map: client idx -> data idxs
    client_data_map = partition_data(global_train_dataset, args, logger)

    # client sampling schedule
    clients_at_rounds = shuffle_clients(args)

    # client_datasets and client_dataloaders
    client_datasets = get_client_datasets(
        global_train_dataset, client_data_map, args)

    global_train_dataloader, global_val_dataloader, global_test_dataloader = get_global_dataloader(
        global_train_dataset, global_val_dataset, global_test_dataset, args)
    client_dataloaders = get_client_dataloaders(client_datasets, args)

    # per-client test dataloaders (subset of global test set by client's classes = For Personalization)
    client_test_dataloaders = get_client_test_dataloaders(
        client_data_map, global_train_dataset, global_test_dataset, args)

    # finetune train dataloaders (do not drop last(smaller than batch_size) = clients having small dataset are included)
    client_finetune_train = {i: torch.utils.data.DataLoader(
        dataset=client_datasets[i], batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=True, num_workers=args.num_workers) for i in range(args.n_clients)}

    return (global_train_dataset, global_val_dataset, global_test_dataset, client_data_map, clients_at_rounds, client_datasets, global_train_dataloader, global_val_dataloader, global_test_dataloader, client_dataloaders, client_test_dataloaders, client_finetune_train)


def get_fedsea_dataloaders(client_datasets, args):
    dataloaders = {}
    datasets = {}

    if args.dataset == 'fmnist':
        normalize = transforms.Normalize(mean=[0.2860], std=[0.3530])

        transform_fedsea = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            normalize
        ])

    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_fedsea = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize
        ])

    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

        transform_fedsea = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize,
        ])

    elif args.dataset == 'tinyimagenet':
        transform_fedsea = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2770, 0.2691, 0.2821)),
        ])

    for i in range(args.n_clients):
        datasets[i] = copy.deepcopy(client_datasets[i])
        datasets[i].transform = transform_fedsea
        client_train_dataloader = data.DataLoader(dataset=datasets[i], batch_size=len(
            datasets[i]), shuffle=False, pin_memory=True, num_workers=args.num_workers)
        dataloaders[i] = client_train_dataloader

    return datasets, dataloaders


def get_client_meta_dataloaders(client_datasets, args):
    dataloaders = {}
    return dataloaders


def init_nets(dataset, num_nets, args, device='cpu', base=False, use_projection_head=False):
    nets = {}

    if hasattr(dataset, 'num_classes'):
        num_classes = dataset.num_classes
    elif hasattr(dataset, 'classes'):
        num_classes = len(dataset.classes)
    else:
        labels = [lbl for _, lbl in getattr(dataset, 'samples', [])]
        num_classes = len(set(labels))

    norm_layer = None
    if args.group_norm:
        def norm_layer(num_channels): return nn.GroupNorm(
            num_groups=args.num_groups, num_channels=num_channels)

    for net_i in range(num_nets):
        if args.model == 'resnet50':
            if base:
                net = resnet_cifar.ResNet50_cifar10(
                    in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer, use_projection_head=use_projection_head)
            else:
                net = resnet_cifar.ResNet50_cifar10(
                    in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer, use_projection_head=False)
        elif args.model == 'resnet18':
            if base:
                net = resnet_cifar.ResNet18_cifar10(
                    in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer, use_projection_head=use_projection_head)
            else:
                net = resnet_cifar.ResNet18_cifar10(
                    in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer, use_projection_head=False)
        elif args.model == 'resnet34':
            if base:
                net = resnet_cifar.ResNet34_cifar10(
                    in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer, use_projection_head=use_projection_head)
            else:
                net = resnet_cifar.ResNet34_cifar10(
                    in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer, use_projection_head=False)
        elif args.model == 'resnet101':
            if base:
                net = resnet_cifar.ResNet101_cifar10(
                    in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer, use_projection_head=use_projection_head)
            else:
                net = resnet_cifar.ResNet101_cifar10(
                    in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer, use_projection_head=False)
        elif args.model == 'resnet152':
            if base:
                net = resnet_cifar.ResNet152_cifar10(
                    in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer, use_projection_head=use_projection_head)
            else:
                net = resnet_cifar.ResNet152_cifar10(
                    in_channels=args.in_channels, num_classes=num_classes, norm_layer=norm_layer, use_projection_head=False)
        elif args.model == 'mobilenet':
            if base:
                net = mobilenet.MobileNetV2(
                    in_channels=args.in_channels, num_classes=num_classes, use_projection_head=use_projection_head)
            else:
                net = mobilenet.MobileNetV2(
                    in_channels=args.in_channels, num_classes=num_classes, use_projection_head=False)
        elif args.model == 'qwen':
            net = QwenLLMWrapper(num_classes=num_classes)
        elif args.model == 'roberta-large':
            from models.roberta_large import RoBERTaLargeWrapper
            net = RoBERTaLargeWrapper(num_classes=num_classes)
        elif args.model in ['roberta-base', 'distilbert', 'llama-3.2-1b']:
            from models.hf_model import HFModelWrapper
            hf_model_name = get_hf_model_name(args.model)
            net = HFModelWrapper(num_classes=num_classes, model_name=hf_model_name)
        elif args.model == 'vit':
            net = ViTWrapper(num_classes=num_classes)
        else:
            supported = ['resnet18', 'resnet34', 'resnet50',
                         'resnet101', 'resnet152', 'mobilenet', 'qwen', 'vit']
            raise ValueError(
                f"wrong model config: '{args.model}'. Supported models are: {supported}")
        net.to(device)
        nets[net_i] = net

    return nets


def compute_accuracy(model, dataloader, device):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if isinstance(x, dict):
                x = {k: v.to(device) for k, v in x.items()}
                target = target.to(dtype=torch.int64).to(device)
                batch_size = x[list(x.keys())[0]].size(0)
            else:
                x, target = x.to(device), target.to(
                    dtype=torch.int64).to(device)
                batch_size = x.data.size(0)
            _, out = model(x)
            _, pred_label = torch.max(out.data, 1)
            total += batch_size
            correct += (pred_label == target.data).sum().item()

    if was_training:
        model.train()

    return correct / float(total)


# Average Accuracy over last n rounds
def avg_last_n(accuracy_list, n):
    if not accuracy_list:
        return None
    recent = accuracy_list[-n:] if len(accuracy_list) >= n else accuracy_list
    return sum(recent) / len(recent)


def get_global_class_center(global_class_center_old, n_party, nets, args, device, train_dataloader):
    DATA_nclass = {'mnist': 10, 'cifar10': 10, 'svhn': 10,
                   'fmnist': 10, 'cifar100': 100, 'tinyimagenet': 200,
                   'ag_news': 4, 'banking77': 77, '20newsgroups': 20,
                   'glue_sst2': 2, 'glue_mnli': 3, 'glue_qnli': 2, 'glue_qqp': 2}
    clsnum = DATA_nclass[args.dataset]
    n_party = int(n_party*args.sample_fraction)
    class_count = torch.zeros((n_party, clsnum), device=device)
    # Shape: [n_party, clsnum, 512]
    class_feature_sum = torch.zeros((n_party, clsnum, 512), device=device)

    for i, (net_id, net) in enumerate(nets.items()):
        net.to(device)
        with torch.no_grad():
            for x, target in (train_dataloader[net_id]):
                if isinstance(x, dict):
                    x = {k: v.to(device) for k, v in x.items()}
                    target = target.to(device)
                else:
                    x, target = x.to(device), target.to(device)
        features, _ = net(x)
        unique_labels, label_counts = torch.unique(
            target, return_counts=True)

        for label, count in zip(unique_labels, label_counts):
            label = int(label.item())
            mask = (target == label)
            class_feature_sum[i, label] += features[mask].sum(0)
            class_count[i, label] += count
    total_class_counts = class_count.sum(dim=0)
    global_center = []

    for cls in range(clsnum):
        centers_sum = class_feature_sum[:, cls].sum(dim=0)

        if total_class_counts[cls] > 0:
            global_center.append(centers_sum / total_class_counts[cls])
        else:
            global_center.append(
                global_class_center_old[cls] if global_class_center_old is not None else torch.zeros(512, device=device))

    return global_center


def init_fedsea(global_model, global_test_dataloader, client_iid_generators, args):
    print('Initialize FedSea')
    dummy_data = next(iter(global_test_dataloader))[0].to(args.device)
    global_model.eval()
    with torch.no_grad():
        feature_dim = global_model(dummy_data)[0].shape[-1]

    print(
        f"Initializing FedSea components with feature dimension: {feature_dim}")

    client_discriminator = ClientDiscriminator(
        feature_dim, args.n_clients).to(args.device)
    optimizer_discriminator = optim.SGD(
        client_discriminator.parameters(), lr=args.lr_server_fedsea)

    attention_p = nn.Parameter(torch.randn(feature_dim, device=args.device))
    optimizer_attention = optim.SGD([attention_p], lr=args.lr_server_fedsea)

    for client_id in range(args.n_clients):
        client_iid_generators[client_id] = IIDFeatureGenerator(
            feature_dim).to(args.device)

    return client_discriminator, optimizer_discriminator, optimizer_attention, attention_p, feature_dim


class DoRASimilarityCalculator:
    def __init__(self, temperature=1.0):
        """
        temperature: KL-Divergence 계산 시 Softmax의 분포를 부드럽게(Smoothing) 조절하는 파라미터
                     값이 클수록 완만한 확률 분포가 생성됩니다.
        """
        self.temperature = temperature

    # ---------------------------------------------------------
    # 1. L2 Distance
    # ---------------------------------------------------------
    def l2_distance_flatten(self, client_tensor, server_tensor):
        """Flatten 방식 L2 거리 계산"""
        return torch.norm(client_tensor.flatten() - server_tensor.flatten(), p=2)

    def l2_distance_filterwise(self, client_tensor, server_tensor):
        """Filter-wise 방식 L2 거리 계산"""
        # Magnitude (m)의 경우 shape이 (out_features, 1) 또는 (out_features,)
        if client_tensor.dim() == 1 or client_tensor.shape[1] == 1:
            # 1D 스칼라들의 차이의 절대값을 구한 뒤 평균
            distances = torch.abs(
                client_tensor.flatten() - server_tensor.flatten())
        else:
            # Direction (V)의 경우 shape이 (out_features, in_features)
            # dim=1 (in_features) 축을 따라 필터별 L2 Norm 계산
            diff = client_tensor - server_tensor
            distances = torch.norm(diff, p=2, dim=1)

        return distances.mean()

    # ---------------------------------------------------------
    # 2. KL-Divergence (KL-Distance)
    # ---------------------------------------------------------
    def kl_distance_flatten(self, client_tensor, server_tensor):
        """Flatten 방식 KL-Divergence (전체 텐서를 하나의 확률 분포로 간주)"""
        # KL Divergence: D_KL(Target || Input). 보통 Server를 Target(정답 확률분포)으로 둡니다.
        server_prob = F.softmax(
            server_tensor.flatten() / self.temperature, dim=0)
        client_log_prob = F.log_softmax(
            client_tensor.flatten() / self.temperature, dim=0)

        # PyTorch F.kl_div는 input이 log-prob, target이 prob 형태여야 함
        kl_div = F.kl_div(client_log_prob, server_prob, reduction='sum')
        return kl_div

    def kl_distance_filterwise(self, client_tensor, server_tensor):
        """Filter-wise 방식 KL-Divergence (필터별 확률 분포 비교)"""
        if client_tensor.dim() == 1 or client_tensor.shape[1] == 1:
            # [주의] Magnitude m은 각 필터가 1개의 스칼라를 가지므로 해당 필터 내에서 분포(Softmax)를 만들 수 없습니다.
            # 따라서 m에 한해서는 필터 간의 '상대적 크기 분포'를 비교하기 위해 Flatten KL로 우회하여 계산합니다.
            return self.kl_distance_flatten(client_tensor, server_tensor)

        # Direction V의 경우: 각 행(필터)마다 in_features 차원(dim=1)을 따라 Softmax 분포 생성
        server_prob = F.softmax(server_tensor / self.temperature, dim=1)
        client_log_prob = F.log_softmax(
            client_tensor / self.temperature, dim=1)

        # reduction='none'으로 텐서 형태 유지 후, 필터(dim=1) 단위로 KL 합산, 이후 전체 필터(dim=0)에 대해 평균
        kl_per_filter = F.kl_div(
            client_log_prob, server_prob, reduction='none').sum(dim=1)

        return kl_per_filter.mean()

    # ---------------------------------------------------------
    # 종합 인터페이스
    # ---------------------------------------------------------
    def compute_dora_divergence(self, m_client, m_server, V_client, V_server, method='l2', mode='filter'):
        """
        크기(m)와 방향(V)에 대한 클라이언트-서버 간 발산(Divergence) 측정
        method: 'l2' 또는 'kl'
        mode: 'flatten' 또는 'filter'
        """
        if method == 'l2':
            if mode == 'flatten':
                dist_m = self.l2_distance_flatten(m_client, m_server)
                dist_V = self.l2_distance_flatten(V_client, V_server)
            else:  # mode == 'filter'
                dist_m = self.l2_distance_filterwise(m_client, m_server)
                dist_V = self.l2_distance_filterwise(V_client, V_server)

        elif method == 'kl':
            if mode == 'flatten':
                dist_m = self.kl_distance_flatten(m_client, m_server)
                dist_V = self.kl_distance_flatten(V_client, V_server)
            else:  # mode == 'filter'
                dist_m = self.kl_distance_filterwise(m_client, m_server)
                dist_V = self.kl_distance_filterwise(V_client, V_server)
        else:
            raise ValueError("method must be 'l2' or 'kl'")

        return dist_m, dist_V
