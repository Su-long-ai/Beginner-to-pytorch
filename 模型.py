import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


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
print(model)
input = torch.ones([64, 3, 32, 32])
output = model(input)
print(output.shape)

writer = SummaryWriter("logs10")
writer.add_graph(model, input)
writer.close()
