from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2 as cv
img_path = "D:\\DataCollect\\train\\ants_image\\24335309_c5ea483bb8.jpg"
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()    # 创建了一个对象
tensor_img = tensor_trans(img)    # ctrl+p可以知道要传入什么类型;这里是将图片转化为tensor(张量/矩阵)类型;这里用了call魔术方法

cv_img = cv.imread(img_path)
cv.imshow("test1", cv_img)
cv.waitKey(100)

writer = SummaryWriter("logs1")
writer.add_image("tensor_img", tensor_img, 1)
writer.close()
