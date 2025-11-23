from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class FCNet_(nn.Module):
    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet_, self).__init__()
        self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)

        self.drop_value = drop
        self.drop = nn.Dropout(drop)

        # in case of using upper character by mistake
        self.activate = activate.lower() if (activate is not None) else None 
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()


    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)
        
        x = self.lin(x)
        
        if self.activate is not None:
            x = self.ac_fn(x)
        return x
 

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
