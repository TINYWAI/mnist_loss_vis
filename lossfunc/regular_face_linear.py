import torch
import torch.nn as nn
import torch.nn.functional as F

class RegularFaceLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(RegularFaceLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        output = F.linear(input, F.normalize(self.weight))

        indices_equal = torch.eye(self.weight.size(0))
        indices_equal = indices_equal.to(self.weight.device)
        mask = 1 - indices_equal

        W_cos_sim_matrix = cos_similarity(self.weight)
        W_cos_sim_matrix = mask * W_cos_sim_matrix
        W_cos_sim = torch.max(W_cos_sim_matrix, 0)[0]
        inter_class_sep = W_cos_sim.mean()

        return output, inter_class_sep


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

def cos_similarity(embeddings):
    """
    cos_dist = <a,b>/(|a||b|)
    """
    # get dot product (batch_size, batch_size)
    dot_product = embeddings.mm(embeddings.t())

    # a vector
    square_sum = dot_product.diag()
    vector_norm = square_sum.sqrt() + 1e-6

    cos_sim = dot_product / vector_norm.unsqueeze(1) / vector_norm.unsqueeze(0)

    return cos_sim