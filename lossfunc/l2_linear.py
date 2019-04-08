import torch
import torch.nn as nn
import torch.nn.functional as F

class L2Linear(nn.Module):
    def __init__(self, in_features, out_features, alpha=10):
        super(L2Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        output = F.linear(self.alpha*F.normalize(input), self.weight)
        # output = F.linear(input, F.normalize(self.weight))

        return output