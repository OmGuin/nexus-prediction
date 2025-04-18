import torch.nn as nn
from config import FEATURES

class IRPredictor(nn.Module):
    def __init__(self, input_dim = len(FEATURES)):
        super(IRPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
        self.net2 = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.Relu(),
            nn.Linear(4, 2),
            nn.Relu(),
            nn.Linear(2,1),
            nn.Relu()
        )

    def forward(self, x):
        return self.net(x)
