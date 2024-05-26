import torchvision
from PIL import Image
from 完整训练模型的model import *

image_path = './dog.png'
image = Image.open(image_path)
image = image.convert('RGB')
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

model = torch.load('model_40.pth', map_location=torch.device('cuda'))
print(model)

image = torch.reshape(image, [1, 3, 32, 32])
image = image.cuda()

model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))
