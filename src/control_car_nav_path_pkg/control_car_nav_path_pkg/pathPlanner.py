# a neural network model to plan path to avoid obstacles and reach target position
# input: current velocity, delta state *10, obstacles *10
# output: planned path points (x, y, angle) *3

import torch.nn as nn

class PathPlanner(nn.Module):
    def __init__(self, input_size=62, hidden_size=128, output_size=9, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, output_size)
        )

        # self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, input):
        outputs = self.model(input)
        # outputs = self.layer_norm(outputs)
        return outputs