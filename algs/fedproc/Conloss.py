from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F




class SupConLoss_new(nn.Module):

    def __init__(self):
        super(SupConLoss_new, self).__init__()

    def forward(self, features, labels, center, args):
        device = args.device

        center = torch.stack(tuple(center), dim=-1).to(device)  # [256, n_classes]
        center = F.normalize(center, p=2, dim=0)

        logits = features @ center

        loss = F.cross_entropy(logits, labels)

        return loss