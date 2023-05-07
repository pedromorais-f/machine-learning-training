from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda image: image / 255)
])

batch_size = 32

data_train = MNIST(root="./", download=True, train=True, transform=transform)
data_test = MNIST(root="./", download=True, train=False, transform=transform)

train_loaded = DataLoader(data_train, batch_size=batch_size,shuffle=True)
test_loaded = DataLoader(data_test, batch_size=batch_size,shuffle=True)


data = iter(train_loaded)
images, labels = next(data)

conv1 = nn.Conv2d(1, 6, 3)
max_pool = nn.MaxPool2d(2)
conv2 = nn.Conv2d(6, 16, 3)

print(images.shape)

x = conv1(images)
print(x.shape)
x = max_pool(x)
print(x.shape)
x = conv2(x)
print(x.shape)
x = max_pool(x)
print(x.shape)
