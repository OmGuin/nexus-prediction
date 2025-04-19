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
        # self.net2 = nn.Sequential(
        #     nn.Linear(input_dim, 4),
        #     nn.ReLU(),
        #     nn.Linear(4, 2),
        #     nn.ReLU(),
        #     nn.Linear(2,1),
        #     nn.ReLU()
        # )
        self.net3 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Identity()
        )

    def forward(self, x):
        return self.net3(x)
