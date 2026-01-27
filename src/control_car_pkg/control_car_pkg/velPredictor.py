# a neural network model using car's current velocity and delta state to predict target velocity

import torch.nn as nn

class VelPredictor(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, output_size=4, dropout=0.2):
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