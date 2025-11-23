import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet


class Classifier(nn.Module):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.lin11 = FCNet(in_features[0], mid_features, activate='relu')
        self.lin12 = FCNet(in_features[1], mid_features, activate='relu')
        self.lin2 = FCNet(mid_features, mid_features, activate='relu')
        self.lin3 = FCNet(mid_features, out_features, drop=drop)

    def forward(self, v, q):
        #x = self.fusion(self.lin11(v), self.lin12(q))
        x = self.lin11(v) * self.lin12(q)
        x = self.lin2(x)
        x = self.lin3(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=False),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits