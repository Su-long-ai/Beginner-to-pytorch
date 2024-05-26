import torch
import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16()

print(vgg16_true)
print(vgg16_false)

train_data = torchvision.datasets.CIFAR10('./dataset', train=True, transform=torchvision.transforms.ToTensor())

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)

# 保存方法一(结构与参数)
torch.save(vgg16_true, 'vgg16_method1.pth')

# 保存方法二(参数)(官方推荐)
torch.save(vgg16_true.state_dict(), 'vgg16_method2.pth')

# 读取方法一
model_1 = torch.load('vgg16_method1.pth')
print(model_1)

# 读取方法二
vgg16_true = torchvision.models.vgg16(weights=False)
vgg16_true.load_state_dict(torch.load('vgg16_method2.pth'))
print(vgg16_true)
