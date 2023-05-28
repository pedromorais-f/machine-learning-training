import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.cnn_layers = nn.Sequential(
            # CONV1
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # CONV2
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(32 * 5 * 5, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def show_image(image):
    plt.figure()
    plt.imshow(image.view(1, 28, 28).detach().cpu().numpy().squeeze())
    plt.show()


# Loading the model
print("Loading the model to test...")
model = torch.load("../Pytorch/Model/model.pt")
model.eval()
print("Model LOADED")

print("Getting the data to test...")
transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])
data_test = MNIST(root="./", download=True, train=False, transform=transform)

print("Loading the test...")
test_loaded = DataLoader(data_test, batch_size=32, shuffle=True)
print("Concluded")

samples = next(iter(test_loaded))
images, labels = samples

print("ALL classes...")
print("0, 1, 2, 3, 4, 5, 6, 7, 8, 9\n")

while True:
    index = input("Put a value(1 - 10000) or exit:")

    if index == "exit":
        break

    index = int(index)
    index -= 1
    actual_number = labels[index].numpy()
    show_image(images[index])
    test_output, last_layer = model(images[index])
    predict = torch.max(test_output, 1)[1].data.numpy().squeeze()

    print(f'Predict: {predict}')
    print(f'Result: {actual_number}\n')
