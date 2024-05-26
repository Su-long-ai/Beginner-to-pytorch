import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d, Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, 64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = MaxPool2d(kernel_size=3, ceil_mode=False)
        self.conv2 = Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv2(x)
        x = self.conv1(self.conv1(self.conv1(x)))
        return x


model = Model()

writer = SummaryWriter("logs7")

step = 0
for data in dataloader:
    imgs, targets = data
    output = model(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
