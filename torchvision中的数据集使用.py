import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 下载数据集和测试数据集(这里使用dataset_transform来使每张图片转化为Tensor类型;这里download为False是因为我已经下载完了)
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=False)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=False)

img, target = test_set[0]
print(test_set[0])
print(test_set.classes)
print(target)
print(test_set.classes[target])
print(img)

writer = SummaryWriter("logs4")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_1", img, i)    # 不能直接test_set[i]，因为test_set[i]有img和target两个
writer.close()
