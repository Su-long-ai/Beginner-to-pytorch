import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, 64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Linear(196608, 10)

    def forward(self, x):
        x = self.conv1(x)
        return x


model = Model()

step = 0
for data in dataloader:
    imgs, targets = data
    output = torch.flatten(imgs)
    output = model(output)


