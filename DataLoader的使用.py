import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor())

# 加载数据的数据集;每个batch加载多少个样本(默认: 1);设置为True时会在每个epoch重新打乱数据(默认: False);
# 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0);
# 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。
# 如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
writer = SummaryWriter("logs5")
step = 0
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("Test_data", imgs, step)
    step = step + 1
writer.close()
