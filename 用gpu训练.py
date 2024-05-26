import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from 完整训练模型的model import *    # *表示全部引入

# 就是模型,数据(输入,标签),损失函数后加cuda()
# 第二种方法
# device = torch.device('cuda')
# xxx = xxx.to(device)    (运用的地方与第一种一样)

writer = SummaryWriter('model')

train_data = torchvision.datasets.CIFAR10('dataset', train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('dataset', train=False, transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)
print('训练数据集的长度为: {}'.format(train_data_size))
print('测试数据集的长度为: {}'.format(test_data_size))

train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

model = Model()
model = model.cuda()

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

learning_rate = 1e-2    # 1*10^(-2)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 100    # 训练次数

total_test_loss = 0
total_accuracy = 0
start_time = time.time()
for i in range(epoch):
    print('第{}次'.format(i+1))

    model.train()    # 设置为训练模式
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        output = model(imgs)
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1

    model.eval()    # 设置为测试模式
    with torch.no_grad():    # 测试
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            output = model(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()
    total_test_step += 1
    end_time = time.time()
    writer.add_scalar('train_loss', total_test_loss, total_test_step)
    writer.add_scalar('train_accuracy', total_accuracy/test_data_size, total_test_step)
    print(total_test_loss)
    print(total_accuracy/test_data_size)
    print(end_time - start_time)
    total_test_loss = 0
    total_accuracy = 0

    torch.save(model, 'model_{}.pth'.format(i), pickle_protocol=4)    # 保存模型

writer.close()
