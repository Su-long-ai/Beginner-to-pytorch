from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
img_path = "D:\\DataCollect\\train\\ants_image\\24335309_c5ea483bb8.jpg"
img_array = np.array(Image.open(img_path))
writer = SummaryWriter("logs")
writer.add_image("test2", img_array, 1, dataformats='HWC')
for i in range(100):
    writer.add_scalar("y = x*x", i*i, i)
writer.close()
