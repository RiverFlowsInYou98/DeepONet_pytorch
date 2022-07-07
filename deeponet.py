from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from FNN import FNN


class DeepONet(nn.Module):

    # Initialize the class
    def __init__(self, layer_size_branch, layer_size_trunk):
        super(DeepONet, self).__init__()

        # initialize parameters and configuration
        self.layer_size_branch = layer_size_branch
        self.layer_size_trunk = layer_size_trunk

        self.loss_fun = self.MSE

        # initialize layers
        self.branch_net = FNN(self.layer_size_branch)
        if callable(self.layer_size_trunk[1]):
            # User-defined trunk net
            self.trunk_net = self.layer_size_trunk[1]
        else:
            self.trunk_net = FNN(self.layer_size_trunk)
        self.bias_last = torch.tebsor(torch.zeros(1), requires_grad=True)


    def forward(self, x_branch, x_trunk):
        # Branch net to encode the input function
        y_branch = self.branch_net(x_branch)
        # Trunk net to encode the domain of the output function
        y_trunk = self.trunk_net(x_trunk)
        # Dot product
        if y_branch.shape[-1] != y_trunk.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        Y = torch.einsum("bi,ni->bn", y_branch, y_trunk)
        # Add bias
        Y += self.bias_last
        return Y

    # mean square error
    def MSE(self, y_true, y_pred):
        return torch.mean(torch.square(y_true - y_pred))
    
    # max L^infinity error of the test data set
    def Max_Linfty_Error(self, y_true, y_pred):
        return torch.max(torch.abs(y_true - y_pred))

    # mean L^infinity error of the test data set
    def Mean_Linfty_Error(self, y_true, y_pred):
        return torch.mean(
            torch.max(torch.abs(y_true - y_pred), dim=1)
        )

    # relative L2 error
    def relative_L2_Error(self, y_true, y_pred):
        return torch.mean(
            torch.norm(y_true - y_pred, dim=1) / torch.norm(y_true, dim=1)
        )

    def get_loss(self, identifier):
        loss_identifier = {
            "mean squared error": self.MSE,
            "MSE": self.MSE,
            "mse": self.MSE,
        }
        if isinstance(identifier, str):
            return loss_identifier[identifier]
        elif callable(identifier):
            return identifier
        else:
            raise ValueError(
                "Could not interpret loss function identifier:", identifier
            )
