from __future__ import division, absolute_import
import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        first_inputs = inputs[:n // 2]
        second_inputs = inputs[n // 2:]
        first_targets = targets[:n // 2]
        second_targets = targets[n // 2:]

        dist = torch.pairwise_distance(first_inputs, second_inputs)
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # print("Distances", dist)
        mask = (first_targets == second_targets).type(torch.cuda.IntTensor)
        # print("Targets", first_targets, second_targets, mask)

        loss_same = mask * dist
        # print("loss_same", loss_same)

        loss_diff = (torch.ones_like(mask) - mask) * dist
        # print(f"Distances same: {mask * dist}, diff: {(torch.ones_like(mask) - mask) * dist}")
        # print("loss_diff", loss_diff)

        # y = torch.ones_like(loss_same)
        return self.ranking_loss(loss_diff, loss_same, torch.ones_like(loss_same))

    # def forward(self, inputs, targets):
    #     """
    #     Args:
    #         inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
    #         targets (torch.LongTensor): ground truth labels with shape (num_classes).
    #     """
    #     n = inputs.size(0)
    #
    #     # Compute pairwise distance, replace by the official when merged
    #     dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    #     dist = dist + dist.t()
    #     dist.addmm_(1, -2, inputs, inputs.t())
    #     dist = dist.clamp(min=1e-12).sqrt() # for numerical stability
    #
    #     # For each anchor, find the hardest positive and negative
    #     mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    #     dist_ap, dist_an = [], []
    #     for i in range(n):
    #         dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
    #         dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
    #     dist_ap = torch.cat(dist_ap)
    #     dist_an = torch.cat(dist_an)
    #
    #     # Compute ranking hinge loss
    #     y = torch.ones_like(dist_an)
    #     return self.ranking_loss(dist_an, dist_ap, y)
