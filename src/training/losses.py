import torch
import torch.nn as nn

class WeightedCrossEntropy(nn.Module):
    """
    Cross-entropy loss with optional class weights.
    """
    def __init__(self, weights=None):
        super(WeightedCrossEntropy, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        return nn.functional.cross_entropy(inputs, targets, weight=self.weights)
