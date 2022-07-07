from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn


class FNN(nn.Module):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes):
        super(FNN, self).__init__()

        self.denses = []
        for i in range(1, len(layer_sizes) - 1):
            self.denses.append(
                nn.Linear(in_features=layer_sizes[i - 1], out_features=layer_sizes[i])
            )
            self.denses.append(nn.ReLU())
        self.denses.append(
            nn.Linear(in_features=layer_sizes[-2], out_features=layer_sizes[-1])
        )

    def forward(self, inputs):
        y = inputs
        for f in self.denses:
            y = f(y)
        return y
