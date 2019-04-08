import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.nn import functional as F

class COCOLogit(nn.Module):
    def __init__(self, feat_dim, num_classes, alpha=6.25):
        super(COCOLogit, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat):
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)
        snfeat = self.alpha * nfeat

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)

        logits = torch.matmul(snfeat, torch.transpose(ncenters, 0, 1))

        # logits = pairwise_distancesAB(feat, self.centers)
        # normC = torch.norm(self.centers, 2, 1)
        # ring_loss = torch.abs(normC - self.alpha).mean()

        return logits

def pairwise_distancesAB(embeddingsA, embeddingsB, squared=False):
    """
    ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2

    input:
    embeddingsA: B x D
    embeddingsB: N x D

    output:
    Eu Distance of size B x N
    """
    # get dot product (batch_size, batch_size)
    dot_product = embeddingsA.mm(embeddingsB.t())

    # a vector
    normA = torch.norm(embeddingsA, 2, 1)
    normB = torch.norm(embeddingsB, 2, 1)

    distances = normA.unsqueeze(1) - 2*dot_product + normB.unsqueeze(0)

    distances = distances.clamp(min=0)

    if not squared:
        epsilon = 1e-16
        mask = torch.eq(distances, 0).float()
        distances = distances + mask * epsilon
        distances = torch.sqrt(distances)
        distances = distances * (1 - mask)

    return distances
