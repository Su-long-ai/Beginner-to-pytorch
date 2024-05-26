from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

writer = SummaryWriter("logs3")

img_path = "D:\\DataCollect\\train\\ants_image\\24335309_c5ea483bb8.jpg"
img = Image.open(img_path)

# ToTensor
trans_totensor = transforms.ToTensor()    # 实例化
tensor_img = trans_totensor(img)
writer.add_image("tensor_img", tensor_img)

# Normalize(归一化)
trans_norm1 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])    # 三个均值和三个标准差，这里假设0.5
norm_img1 = trans_norm1(tensor_img)
writer.add_image("norm_img", norm_img1, 1)
trans_norm2 = transforms.Normalize([6, 5, 4], [1, 2, 3])    # 三个均值和三个标准差，这里假设0.5
norm_img2 = trans_norm2(tensor_img)
writer.add_image("norm_img", norm_img2, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)    # 在上面实例化过了，这里转化为totensor类型(原PIL)
writer.add_image("Resize", img_resize, 1)
print(img_resize)

# Compose
trans_resize_2 = transforms.Resize(512)    # 参数为一个时，最短边的像素点个数
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 3)

# RandomCrop
trans_random = transforms.RandomCrop(100)
for i in range(10):
    img_random = trans_random(img)
    img_random = trans_totensor(img_random)
    writer.add_image("Random_img_2", img_random, i)
'''    # 另一种方法，实际上功能一样
trans_random = transforms.RandomCrop(100)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_random = trans_compose_2(img)
    writer.add_image("Random_img", img_random, i)
'''

writer.close()
