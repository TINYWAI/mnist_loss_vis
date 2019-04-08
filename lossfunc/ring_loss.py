import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.nn import functional as F

class RingLoss(nn.Module):
    """
    Refer to paper
    Ring loss: Convex Feature Normalization for Face Recognition
    """
    def __init__(self, type='L2'):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.type = type

    def forward(self, x):
        # print(self.radius)
        x = x.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] < 0: # Initialize the radius with the mean feature norm of first iteration
            self.radius.data.fill_(x.mean().item())
        if self.type == 'L1': # Smooth L1 Loss
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x))
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x)
            ringloss = loss1 + loss2
        elif self.type == 'auto': # Divide the L2 Loss by the feature's own norm
            diff = x.sub(self.radius.expand_as(x)) / (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq
        else: # L2 Loss, if not specified
            diff = x.sub(self.radius.expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq
        return ringloss

def cos_distances(embeddings):
    """
    cos_dist = 1 - <a,b>/(|a||b|)
    """
    # get dot product (batch_size, batch_size)
    dot_product = embeddings.mm(embeddings.t())

    # a vector
    square_sum = dot_product.diag()
    vector_norm = square_sum.sqrt() + 1e-6

    cos_sim = dot_product / vector_norm.unsqueeze(1) / vector_norm.unsqueeze(0)
    distances = 1 - cos_sim
    distances = distances.clamp(min=0)

    return distances


def pairwise_distances(embeddings, squared=False):
    """
    ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
    """
    # get dot product (batch_size, batch_size)
    dot_product = embeddings.mm(embeddings.t())

    # a vector
    square_sum = dot_product.diag()

    distances = square_sum.unsqueeze(1) - 2*dot_product + square_sum.unsqueeze(0)

    distances = distances.clamp(min=0)

    if not squared:
        epsilon = 1e-16
        mask = torch.eq(distances, 0).float()
        distances = distances + mask * epsilon
        distances = torch.sqrt(distances)
        distances = distances * (1 - mask)

    return distances
