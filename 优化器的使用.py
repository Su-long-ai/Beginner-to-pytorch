import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, 64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.conv_1 = Conv2d(3, 32, 5, padding=2)
        # self.conv_2 = Conv2d(32, 32, 5, padding=2)
        # self.conv_3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool_1 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear_1 = Linear(1024, 64)
        # self.linear_2 = Linear(64, 10)

        self.model_1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        # x = self.conv_1(x)
        # x = self.maxpool_1(x)
        # x = self.conv_2(x)
        # x = self.maxpool_1(x)
        # x = self.conv_3(x)
        # x = self.maxpool_1(x)
        # x = self.flatten(x)
        # x = self.linear_1(x)
        # x = self.linear_2(x)
        x = self.model_1(x)
        return x


model = Model()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), 0.1)
for epoch in range(2000):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = model(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss
    print(running_loss)
