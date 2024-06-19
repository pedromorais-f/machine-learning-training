from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import mlflow


transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

data_train = MNIST(root="./", download=True, train=True, transform=transform)
data_test = MNIST(root="./", download=True, train=False, transform=transform)

#-------------------------------------------------------------------------------

from torch.utils.data import DataLoader

train_loaded = DataLoader(data_train, batch_size=32, shuffle=True)
test_loaded = DataLoader(data_test, batch_size=32, shuffle=True)

#-------------------------------------------------------------------------------

from torch import nn

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.cnn_layers = nn.Sequential(
            #CONV1
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            #CONV2
            nn.Conv2d(6, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.linear_layers = nn.Sequential(
            nn.Linear(32*5*5, 200),
            nn.ReLU(),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Linear(80, 10),
        )
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    

model = MnistModel()
#-------------------------------------------------------------------------------

params = {
    "learning-rate": 0.001,
    "epochs": 5
}

#Defining an optimizer and loss
from torch import optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#-------------------------------------------------------------------------------

#Training the model
from tqdm import tqdm
import torch
#Lists to get all the data about training
train_loss, test_loss = [], []
accuracy_train, accuracy_test = [], []
for epoch in range(params["epochs"]):
    total_train_loss = 0
    total_test_loss = 0
    
    model.train()
    
    total = 0
    for index, (image, label) in tqdm(enumerate(train_loaded), desc=f"Fitting Epoch {epoch + 1}"):
        
        optimizer.zero_grad()
        
        pred = model(image)
        
        loss = criterion(pred, label)
        total_train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        pred = nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1
    
    train_accuracy = total / len(data_train)
    total_train_loss = total_train_loss / (index + 1)
    
    accuracy_train.append(train_accuracy)
    train_loss.append(total_train_loss)
    
    #Validating the model
    model.eval()
    total = 0
    for index, (image, label) in tqdm(enumerate(test_loaded), desc="Validating the model"):
        pred = model(image)
        
        loss = criterion(pred, label)
        total_test_loss += loss.item()
        
        pred = nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1
    test_accuracy = total / len(data_test)
    total_test_loss = total_test_loss / (index + 1)
    
    accuracy_test.append(test_accuracy)
    test_loss.append(total_test_loss)
    
    print("Epoch: {}/{}  ".format(epoch + 1, params["epochs"]),
            "Training loss: {:.4f}  ".format(total_train_loss),
            "Testing loss: {:.4f}  ".format(total_test_loss),
            "Train accuracy: {:.4f}  ".format(train_accuracy),
            "Test accuracy: {:.4f}  ".format(test_accuracy))

#-------------------------------------------------------------------------------
mlflow.set_tracking_uri(uri="")

mlflow.set_experiment("MLFlow Start")

with mlflow.start_run():
    mlflow.log_params(params)

    for idx, acc in enumerate(accuracy_test):
        mlflow.log_metric(f"accuracy {idx}", acc)

    model_info = mlflow.pytorch.log_model(
        pytorch_model= model,
        artifact_path = "mnist_model"
    )