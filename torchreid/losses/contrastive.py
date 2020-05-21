from __future__ import division, absolute_import
import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """Contrastive loss.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, mode="adaptive"):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.weight_diff_min = 0.2  # Minimal weight for term of loss related to different ids
        self.mode = mode

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

        loss_diff = (torch.ones_like(mask) - mask) * torch.max((torch.full_like(dist, fill_value=self.margin) - dist),
                                                               torch.zeros_like(dist))
        # print(f"Distances same: {mask * dist}, diff: {(torch.ones_like(mask) - mask) * dist}")
        # print("loss_diff", loss_diff)

        return loss_same.sum() + loss_diff.sum()

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
    #     dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    #
    #     # Create matrix telling whether calculated distance is between objects of the same id
    #     mask = (targets.expand(n, n).eq(targets.expand(n, n).t())).type(torch.cuda.IntTensor)
    #     print(f"targets: {targets}")
    #     print(f"Mask {mask}")
    #     if self.mode == 'adaptive':
    #         # Since number of objects with the same and with different ids are different (there are way more objects of
    #         # different id) we create weights proportional to number of instances of particular class (same id, different id)
    #         n_same_ids = (mask.sum().item() - n) / 2  # Number of different objects with the same id
    #         n_dist = (n * n - n) / 2  # Total number of distances between different objects
    #         print(f"Same ids: {n_same_ids}, n_dist: {n_dist}")
    #         weight_diff = max(n_same_ids / n_dist, self.weight_diff_min)
    #         weight_same = 1 - weight_diff
    #     elif self.mode == 'const':
    #         weight_same = 1
    #         weight_diff = 10
    #     else:
    #         raise NotImplementedError
    #     print(f"weight_same: {weight_same}, weight_diff: {weight_diff}")
    #     dist_same = (mask - torch.eye(n).cuda()) * dist  # Not counting distances between same objects
    #     dist_diff = (torch.ones_like(mask) - mask) * torch.max((torch.full_like(dist, fill_value=self.margin) - dist),
    #                                                            torch.zeros_like(dist))
    #     # print("dist_same\n", dist_same)
    #     # print("dist_diff\n", dist_diff)
    #     loss = weight_same * dist_same.sum() + weight_diff * dist_diff.sum()
    #     # print("loss_same matrix \n", weight_same * dist_same)
    #     # print("loss_same\n", weight_same * dist_same.sum())
    #     # print("loss_diff matrix\n", weight_diff * dist_diff)
    #     # print("loss_diff\n", weight_diff * dist_diff.sum())
    #     # print("Total loss", loss, f"Same ids: {n_same_ids}")
    #
    #     return loss
