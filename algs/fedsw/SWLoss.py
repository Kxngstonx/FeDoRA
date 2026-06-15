import torch
import torch.nn as nn
import torch.nn.functional as F






class SampleWiseLoss(nn.Module):

    def __init__(self):
        super(SampleWiseLoss, self).__init__()

    def forward(self, cosine_sim_matrix, x, target, args):
        target_expand = target.unsqueeze(1)
        mask_same = (target_expand == target_expand.t())
        valid_mask = ~torch.eye(x.size(0), dtype=torch.bool, device=args.device)
        mask_same = mask_same & valid_mask
        mask_diff = (target_expand != target_expand.t())

        pos_sim = torch.where(mask_same, cosine_sim_matrix, torch.tensor(2.0, device=args.device))
        hard_positive, _ = pos_sim.min(dim=1)
        
        pos_sim_max = torch.where(mask_same, cosine_sim_matrix, torch.tensor(-2.0, device=args.device))
        easy_positive, _ = pos_sim_max.max(dim=1)
        
        neg_sim = torch.where(mask_diff, cosine_sim_matrix, torch.tensor(2.0, device=args.device))
        hard_negative, _ = neg_sim.min(dim=1)
        
        loss_pull = F.binary_cross_entropy_with_logits(hard_positive, torch.ones_like(hard_positive))
        loss_push_pos = F.binary_cross_entropy_with_logits(easy_positive, torch.zeros_like(easy_positive))
        loss_push_neg = F.binary_cross_entropy_with_logits(hard_negative, torch.zeros_like(hard_negative))
        loss_push = loss_push_pos + loss_push_neg
        
        sw_loss = args.lambda_pull * loss_pull + args.lambda_push * loss_push

        return sw_loss
