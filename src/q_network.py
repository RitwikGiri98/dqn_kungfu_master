# src/q_network.py
import torch, torch.nn as nn

class DQN(nn.Module):
    def __init__(self, in_ch, n_actions):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),  nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64*7*7, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    def forward(self, x): return self.head(self.body(x))
