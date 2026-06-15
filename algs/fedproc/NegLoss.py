from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F




class NegLoss_new(nn.Module):
    def __init__(self):
        super(NegLoss_new, self).__init__()

    def forward(self, features, labels, center, args):
        DATA_nclass = {'mnist': 10, 'cifar10': 10, 'svhn': 10, 'fmnist': 10, 'cifar100': 100, 'tinyimagenet': 200}
        device = args.device
        Data_nclasses = DATA_nclass[args.dataset]

        center = torch.stack(center, dim=1).to(device)
        center = F.normalize(center, p=2, dim=0)

        batch_size = features.shape[0]
        all_classes = torch.arange(Data_nclasses, device=device).unsqueeze(0)
        labels = labels.unsqueeze(1)

        different_labels = all_classes.repeat(batch_size, 1)
        mask = different_labels != labels

        current_centers = center.T.unsqueeze(0).repeat(batch_size, 1, 1)
        masked_centers = current_centers[mask].view(batch_size, Data_nclasses - 1, -1)

        features = F.normalize(features, p=2, dim=1)
        features = features.unsqueeze(1)
        
        dot_products = torch.bmm(features, masked_centers.permute(0, 2, 1)).squeeze(1)
        
        neg_list = dot_products.mean(dim=1).unsqueeze(1)
        
        neg_labels = torch.ones(batch_size, 1, device=device)
        
        logits = torch.cat((neg_list, neg_labels), dim=1)
        
        return logits




# from __future__ import print_function

# import torch
# import torch.nn as nn
# import torch.nn.functional as F



    
# class NegLoss_new(nn.Module):

#     def __init__(self):
#         super(NegLoss_new, self).__init__()

#     def forward(self, features, labels, center, args):
#         DATA_nclass = {'mnist': 10, 'cifar10': 10, 'svhn': 10, 'fmnist': 10, 'cifar100': 100, 'tinyimagenet': 200}
#         device = args.device
#         Data_nclasses = DATA_nclass[args.dataset]

#         center = torch.stack(center, dim=1).to(device)  # [256, n_classes]
#         center = F.normalize(center, p=2, dim=0)  # [256, n_classes]

#         batch_size = features.shape[0]
#         all_classes = torch.arange(Data_nclasses, device=device).unsqueeze(0)  # [1, n_classes]
#         labels = labels.unsqueeze(1)  # [batch_size, 1]

#         different_labels = all_classes.repeat(batch_size, 1)  # [batch_size, n_classes]
#         mask = different_labels != labels  # [batch_size, n_classes]

#         current_centers = center.T.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, n_classes, 256]
#         masked_centers = current_centers[mask].view(batch_size, Data_nclasses - 1, -1)  # [batch_size, n_classes-1, 256]

#         features = features.unsqueeze(1)
#         dot_products = torch.bmm(features, masked_centers.permute(0, 2, 1)).squeeze(1)  # [batch_size, n_classes-1]

#         neg_list = dot_products.max(dim=1)[0].unsqueeze(1)  # [batch_size, 1]

#         neg_labels = torch.ones(batch_size, 1, device=device)

#         logits = torch.cat((neg_list, neg_labels), dim=1)  # [batch_size, 2]

#         return logits
