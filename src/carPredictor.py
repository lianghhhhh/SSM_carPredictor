# state space model based neural network model for car price prediction
# input: u & y, output: 4 matrix A, B, C, D defining the state space model
import torch
import torch.nn as nn

class CarPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=25, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)