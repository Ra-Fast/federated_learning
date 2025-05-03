"""my-awesome-app: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize


class Net(nn.Module):

    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 5 * 5)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     return self.fc3(x)
    # def __init__(self):
    #     super(Net, self).__init__()
    #     # Bloque convolucional 1
    #     # Input: 3x32x32
    #     self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    #     self.bn1 = nn.BatchNorm2d(32)
    #     # Output: 32x32x32
    #     self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    #     # Output: 32x16x16

    #     # Bloque convolucional 2
    #     self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    #     self.bn2 = nn.BatchNorm2d(64)
    #     # Output: 64x16x16
    #     self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    #     # Output: 64x8x8

    #     # Bloque convolucional 3
    #     self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    #     self.bn3 = nn.BatchNorm2d(128)
    #     # Output: 128x8x8
    #     self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    #     # Output: 128x4x4

    #     # Capas totalmente conectadas (clasificador)
    #     # Aplanar la salida del último bloque convolucional
    #     # El tamaño es out_channels * height * width = 128 * 4 * 4 = 2048
    #     self.fc1 = nn.Linear(128 * 4 * 4, 512)
    #     self.dropout1 = nn.Dropout(0.5) # Dropout para regularización
    #     self.fc2 = nn.Linear(512, 128)
    #     self.dropout2 = nn.Dropout(0.5) # Más regularización
    #     self.fc3 = nn.Linear(128, 10) # Salida final para las clases

    # def forward(self, x):
    #     # Bloque 1
    #     x = self.pool1(F.relu(self.bn1(self.conv1(x))))
    #     # Bloque 2
    #     x = self.pool2(F.relu(self.bn2(self.conv2(x))))
    #     # Bloque 3
    #     x = self.pool3(F.relu(self.bn3(self.conv3(x))))

    #     # Aplanar antes de las capas FC
    #     # El view se adapta al tamaño calculado: 128 * 4 * 4
    #     x = x.view(-1, 128 * 4 * 4)

    #     # Clasificador
    #     x = self.dropout1(F.relu(self.fc1(x)))
    #     x = self.dropout2(F.relu(self.fc2(x)))
    #     x = self.fc3(x) # No se aplica ReLU ni Softmax aquí, CrossEntropyLoss lo incluye
    #     return x
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.6)
        self.fc3 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def get_transforms():
        
    pytorch_transforms = Compose([
            Resize((32, 32)),                      # Cambia el tamaño a 32x32 píxeles
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch
    
    return apply_transforms

fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int):
    """Load partition RaFast/RepoMED data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        #partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label", alpha=1.0)
        partitioner=IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="RaFast/RepoMED_1",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    partition_train_test = partition_train_test.with_transform(get_transforms())
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
