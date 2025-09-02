import torch.nn as nn
import torch
device = torch.device("cpu")

    # model.py
class ResPINN(nn.Module):
    def __init__(self, num_hidden_layers=12, num_neurons=128):
        super(ResPINN, self).__init__()
        self.num_neurons = num_neurons
        self.input_layer = nn.Linear(6, num_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_neurons, num_neurons),
                nn.Tanh()
            ) for _ in range(num_hidden_layers)
        ])
        self.output_layer = nn.Linear(num_neurons, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            x = x + residual
        return self.output_layer(x)