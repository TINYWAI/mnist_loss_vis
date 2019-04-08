import torch
import torch.nn as nn
import torch.nn.functional as F

class ReciprocalNormLoss(nn.Module):
    """
    Refer to paper
    Ring loss: Convex Feature Normalization for Face Recognition
    """
    def __init__(self, eps=1e-6):
        super(ReciprocalNormLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        nfeats = torch.norm(x, 2, 1) + self.eps
        reciprocal_nfeats = 1 / nfeats
        loss = reciprocal_nfeats.mean()

        return loss