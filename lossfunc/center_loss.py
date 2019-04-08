import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.nn import functional as F


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, sphere_type=None, R=0, size_average=True, metric_mode='pairwise', force=False):
        super(CenterLoss, self).__init__()
        if R != 0:
            self.centers = nn.Parameter(R * torch.randn(num_classes, feat_dim))
        else:
            self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.metric_mode = metric_mode
        self.sphere_type = sphere_type
        self.R = R
        if self.metric_mode == 'pairwise':
            self.centerlossfunc = CenterlossFunc.apply
        elif self.metric_mode == 'cos':
            self.centerlossfunc = CosCenterlossFunc.apply
        # self.cos_centerlossfunc = CosCenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average
        self.force = force

    def forward(self, feat, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        # feat = feat / torch.norm(feat, 2, 1).unsqueeze(1)
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)

        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return -grad_output * diff / batch_size, None, grad_centers / batch_size, None

class CosCenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        # cos_distance = (feature - centers_batch).pow(2).sum() / 2.0 / batch_size
        cos_distance = (1 - F.cosine_similarity(feature, centers_batch)).sum() / batch_size
        return cos_distance

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        # diff = centers_batch - feature
        norm_feat_centers =  torch.norm(feature, 2, 1).unsqueeze(1) * torch.norm(centers_batch, 2, 1).unsqueeze(1)
        diff_feat = -centers_batch / norm_feat_centers
        diff_centers = -feature / norm_feat_centers
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff_centers)
        grad_centers = grad_centers/counts.view(-1, 1)
        return grad_output * diff_feat / batch_size, None, grad_centers / batch_size, None


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
