import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x + 1
        return x


model = Model()
x = torch.tensor(1.0)
output = model(x)
print(output)
