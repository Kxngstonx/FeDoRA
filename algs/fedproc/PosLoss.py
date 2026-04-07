import torch
import torch.nn as nn
import torch.nn.functional as F






class PosLoss_new(nn.Module):

    def __init__(self):
        super(PosLoss_new, self).__init__()

    def forward(self, features, labels, center, args):
        device = args.device
        
        center = torch.stack(center, dim=-1).to(device)  # [256, n_classes]
        center = F.normalize(center, p=2, dim=0)  # [256, n_classes]

        batch_centers = center[:, labels].T  # [batch_size, 256]

        features = F.normalize(features, p=2, dim=1)  # [batch_size, 256]
        pos_list = torch.sum(features * batch_centers, dim=1)  # [batch_size]

        logits = torch.cat((pos_list.unsqueeze(1), torch.ones_like(pos_list.unsqueeze(1))), dim=1)  # [batch_size, 2]

        return logits
